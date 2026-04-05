import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzIconModule } from 'ng-zorro-antd/icon';
import { NzSwitchModule } from 'ng-zorro-antd/switch';
import { FormsModule } from '@angular/forms';
import { HighlightDirective } from './highlight.directive';
import { UnlessDirective } from './unless.directive';
import { TimeDisplayAttrDirective } from './time-display-attr.directive';
import { IfAdminDirective } from './if-admin.directive';

@Component({
  selector: 'app-Directives',
  standalone: true,
  imports: [
    CommonModule,
    HighlightDirective,
    UnlessDirective,
    TimeDisplayAttrDirective,
    IfAdminDirective,
    RouterLink,
    NzButtonModule,
    NzIconModule,
    NzSwitchModule,
    FormsModule,
  ],
  templateUrl: './directives.component.html',
  styleUrl: './directives.component.css',
})
export class DirectivesComponent implements OnInit {
  // 用于演示结构型指令的变量
  showContent = true;
  isAdmin = true;
  timeFormat = '12';

  constructor() {}

  ngOnInit(): void {}

  toggleContent() {
    this.showContent = !this.showContent;
  }

  toggleAdmin() {
    this.isAdmin = !this.isAdmin;
  }

  toggleTimeFormat() {
    this.timeFormat = this.timeFormat === '12' ? '24' : '12';
  }
}
