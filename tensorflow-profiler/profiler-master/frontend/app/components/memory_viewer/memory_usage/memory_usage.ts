import {MemoryViewerPreprocessResult} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {HeapObject} from 'org_xprof/frontend/app/common/interfaces/heap_object';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';

interface MemoryUsageBytes {
  padded: number;
  unpadded: number;
}

/**
 * Provides calculation of memory usage from xla buffer assignment.
 * @final
 */
export class MemoryUsage {
  private nColor: number;

  peakHeapSizeBytes: number;
  paddingOverhead: number;
  hloTempSizeBytes: number;
  hloTempFragmentation: number;
  totalArgumentSizeBytes: number;
  peakLogicalBuffers: number[];
  peakHeapSizePosition: number;
  indefiniteMemoryUsageBytes: MemoryUsageBytes;
  heapSizes: number[];
  unpaddedHeapSizes: number[];
  maxHeap: HeapObject[];
  maxHeapBySize: HeapObject[];
  maxHeapByPaddingSize: HeapObject[];
  bySizeToMaxHeap: number[];
  maxHeapToBySize: number[];
  byPaddingSizeToMaxHeap: number[];
  maxHeapToByPaddingSize: number[];
  logicalBufferSpans: {[key: number]: number[]};

  smallBufferSize: number;

  diagnostics: {errors: string[], warnings: string[], info: string[]};
  moduleName: string;
  timelineUrl: string;

  // Only one of hloProto or preprocess is valid to construct MemoryUsage.
  constructor(
      preprocess: MemoryViewerPreprocessResult|null, memorySpaceColor: number) {
    this.nColor = 0;

    this.peakHeapSizeBytes = 0;
    this.paddingOverhead = 0;
    this.hloTempSizeBytes = 0;
    this.hloTempFragmentation = 0;
    this.totalArgumentSizeBytes = 0;
    this.peakLogicalBuffers = [];
    this.peakHeapSizePosition = 0;
    this.indefiniteMemoryUsageBytes = {padded: 0, unpadded: 0};
    this.heapSizes = [];
    this.unpaddedHeapSizes = [];
    this.maxHeap = [];
    this.maxHeapBySize = [];
    this.maxHeapByPaddingSize = [];
    this.bySizeToMaxHeap = [];
    this.maxHeapToBySize = [];
    this.byPaddingSizeToMaxHeap = [];
    this.maxHeapToByPaddingSize = [];
    this.logicalBufferSpans = {};
    this.smallBufferSize = 16 * 1024;
    this.diagnostics = {errors: [], warnings: [], info: []};
    this.moduleName = '';
    this.timelineUrl = '';

    // Both input sources (HLOProto and preprocessed data) are invalid.
    if (!preprocess) {
      this.diagnostics.errors.push(
          'We failed to fetch a valid input. The input is empty or too large.');
      return;
    }

    if (preprocess) {
      // Initialize memory viewer from preprocessed data.
      this.initMemoryUsageFromPrecomputed(preprocess);
    }
  }

  /**
   * Initializes memory usage from precomputed results.
   */
  private initMemoryUsageFromPrecomputed(preprocess:
                                             MemoryViewerPreprocessResult) {
    // Copy the fields from preprocessed result.
    this.moduleName = preprocess.moduleName || '';
    this.timelineUrl = preprocess.allocationTimeline || '';
    if (!this.timelineUrl.startsWith('/memory_viewer.json')) {
      this.timelineUrl = '';
    }
    this.peakHeapSizeBytes =
        (preprocess.totalBufferAllocationMib || 0) * 1024 * 1024;
    this.paddingOverhead = (preprocess.peakHeapMib || 0) * 1024 * 1024 -
        (preprocess.peakUnpaddedHeapMib || 0) * 1024 * 1024;
    this.totalArgumentSizeBytes =
        (preprocess.entryComputationParametersMib || 0) * 1024 * 1024;
    this.hloTempSizeBytes = this.peakHeapSizeBytes -
        (preprocess.indefiniteBufferAllocationMib || 0) * 1024 * 1024;
    const framenentationSizeBytes =
        this.peakHeapSizeBytes - (preprocess.peakHeapMib || 0) * 1024 * 1024;
    if (this.hloTempSizeBytes) {
      this.hloTempFragmentation =
          framenentationSizeBytes / this.hloTempSizeBytes;
    }

    this.peakHeapSizePosition = (preprocess.peakHeapSizePosition || 0);
    this.heapSizes = preprocess.heapSizes || [];
    this.unpaddedHeapSizes = preprocess.unpaddedHeapSizes || [];
    if (preprocess.logicalBufferSpans) {
      for (const [key, value] of Object.entries(
               preprocess.logicalBufferSpans)) {
        this.logicalBufferSpans[utils.toNumber(key)] =
            [value.start || 0, value.limit || 0];
      }
    }
    for (const heapObject of preprocess.maxHeap || []) {
      this.maxHeap.push({
        instructionName: heapObject.instructionName,
        shape: heapObject.shapeString,
        tfOpName: heapObject.tfOpName,
        sizeMiB: heapObject.logicalBufferSizeMib,
        unpaddedSizeMiB: heapObject.unpaddedShapeMib,
        color: this.nColor++,
        groupName: heapObject.groupName,
        opcode: heapObject.opCode,
        logicalBufferId: heapObject.logicalBufferId,
      });
    }
    this.createMaxHeapIndex();
  }

  /**
   * Create index for this.maxHeap so it can be selected by size, unpadded size
   * and etc.
   */
  private createMaxHeapIndex() {
    const indexedMaxHeap = this.maxHeap.map((e, i) => {
      return {ind: i, val: e};
    });
    // Sort max heap objects by size and create index.
    indexedMaxHeap.sort((a, b) => {
      const sizeA = a && a.val && a.val.sizeMiB ? a.val.sizeMiB : 0;
      const sizeB = b && b.val && b.val.sizeMiB ? b.val.sizeMiB : 0;
      return sizeB - sizeA;
    });
    this.maxHeapBySize = indexedMaxHeap.map(e => e.val);
    this.bySizeToMaxHeap = indexedMaxHeap.map(e => e.ind);
    this.maxHeapToBySize.length = this.maxHeap.length;
    for (let i = 0; i < this.bySizeToMaxHeap.length; i++) {
      this.maxHeapToBySize[this.bySizeToMaxHeap[i]] = i;
    }
    // Sort max heap objects by padding size and create index.
    indexedMaxHeap.sort((a, b) => {
      const paddingA = a && a.val && a.val.sizeMiB && a.val.unpaddedSizeMiB ?
          a.val.sizeMiB - a.val.unpaddedSizeMiB :
          0;
      const paddingB = b && b.val && b.val.sizeMiB && b.val.unpaddedSizeMiB ?
          b.val.sizeMiB - b.val.unpaddedSizeMiB :
          0;
      return paddingB - paddingA;
    });
    this.maxHeapByPaddingSize = indexedMaxHeap.map(e => e.val);
    this.byPaddingSizeToMaxHeap = indexedMaxHeap.map(e => e.ind);
    this.maxHeapToByPaddingSize.length = this.maxHeap.length;
    for (let i = 0; i < this.byPaddingSizeToMaxHeap.length; i++) {
      this.maxHeapToByPaddingSize[this.byPaddingSizeToMaxHeap[i]] = i;
    }
  }
}
