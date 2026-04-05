// time-display-attr.directive.ts - 时间显示属性型指令
import {
  Directive,
  Input,
  ElementRef,
  Renderer2,
  OnDestroy,
  OnInit,
  OnChanges,
} from '@angular/core';

@Directive({
  selector: '[appTimeDisplayAttr]',
})
export class TimeDisplayAttrDirective implements OnInit, OnDestroy, OnChanges {
  private intervalId: any;

  @Input() appTimeDisplayAttr: string = '12'; // 12小时制或24小时制
  @Input() updateInterval: number = 1000; // 更新间隔，毫秒

  // 监听输入变化
  ngOnChanges() {
    if (this.intervalId) {
      this.updateTime();
    }
  }

  constructor(
    private elementRef: ElementRef,
    private renderer: Renderer2
  ) {}

  ngOnInit() {
    this.startTimeDisplay();
  }

  ngOnDestroy() {
    this.stopTimeDisplay();
  }

  private startTimeDisplay() {
    this.updateTime();
    this.intervalId = setInterval(() => {
      this.updateTime();
    }, this.updateInterval);
  }

  private stopTimeDisplay() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
  }

  private updateTime() {
    const now = new Date();
    let timeString: string;

    if (this.appTimeDisplayAttr === '24') {
      timeString = now.toLocaleTimeString('zh-CN', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } else {
      timeString = now.toLocaleTimeString('zh-CN', {
        hour12: true,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    }

    // 更新元素内容
    this.renderer.setProperty(this.elementRef.nativeElement, 'textContent', timeString);
  }
}
