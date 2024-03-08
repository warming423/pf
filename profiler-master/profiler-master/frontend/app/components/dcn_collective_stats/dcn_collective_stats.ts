import {Component, OnDestroy} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {TABLE_OPTIONS} from 'org_xprof/frontend/app/components/chart/chart_options';
import {Dashboard} from 'org_xprof/frontend/app/components/chart/dashboard/dashboard';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {DataService} from 'org_xprof/frontend/app/services/data_service/data_service';
import {setLoadingStateAction} from 'org_xprof/frontend/app/store/actions';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

const DCN_COLLECTIVE_STATS_INDEX = 0;

/** A Dcn Collective Stats page component. */
@Component({
  selector: 'dcn-collective-stats',
  templateUrl: './dcn_collective_stats.ng.html',
  styleUrls: ['./dcn_collective_stats.scss']
})
export class DcnCollectiveStats extends Dashboard implements OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);

  dataInfo: ChartDataInfo = {
    data: null,
    dataProvider: new DefaultDataProvider(),
    filters: [],
    options: {
      ...TABLE_OPTIONS,
      showRowNumber: false,
    },
  };

  constructor(
      route: ActivatedRoute, private readonly dataService: DataService,
      private readonly store: Store<{}>) {
    super();
    route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.update(params as NavigationEvent);
    });
  }

  update(event: NavigationEvent) {
    const run = event.run || '';
    const tag = event.tag || 'dcn_collective_stats';
    const host = event.host || '';

    this.store.dispatch(setLoadingStateAction({
      loadingState: {
        loading: true,
        message: 'Loading Dcn Collective Stats data',
      }
    }));

    this.dataService.getData(run, tag, host)
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          this.store.dispatch(setLoadingStateAction({
            loadingState: {
              loading: false,
              message: '',
            }
          }));

          const d = data as SimpleDataTable[] | null;
          if (d && d.hasOwnProperty(DCN_COLLECTIVE_STATS_INDEX)) {
            this.parseData(d[DCN_COLLECTIVE_STATS_INDEX]);
            this.dataInfo = {
              ...this.dataInfo,
              data: d[DCN_COLLECTIVE_STATS_INDEX],
            };
          }
        });
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
