import {NgModule} from '@angular/core';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';

import {ProgramOrderChart} from './program_order_chart';

@NgModule({
  declarations: [ProgramOrderChart],
  imports: [
    ChartModule,
  ],
  exports: [ProgramOrderChart]
})
export class ProgramOrderChartModule {
}
