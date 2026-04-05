// time-display.directive.ts - 时间显示结构型指令
import {
  Directive,
  Input,
  TemplateRef,
  ViewContainerRef,
  OnDestroy,
  OnInit,
  ElementRef,
  Renderer2,
} from '@angular/core';

@Directive({
  selector: '[appTimeDisplay]',
})
export class TimeDisplayDirective implements OnInit, OnDestroy {
  private intervalId: any;
  private hasView = false;

  @Input() appTimeDisplay: string = '12'; // 12小时制或24小时制
  @Input() updateInterval: number = 1000; // 更新间隔，毫秒

  constructor(
    private templateRef: TemplateRef<any>,
    private viewContainer: ViewContainerRef,
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
    if (!this.hasView) {
      this.viewContainer.createEmbeddedView(this.templateRef);
      this.hasView = true;
    }

    this.updateTime();
    this.intervalId = setInterval(() => {
      this.updateTime();
    }, this.updateInterval);
  }

  private stopTimeDisplay() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
    if (this.hasView) {
      this.viewContainer.clear();
      this.hasView = false;
    }
  }

  private updateTime() {
    const now = new Date();
    let timeString: string;

    if (this.appTimeDisplay === '24') {
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
    const element = this.elementRef.nativeElement;
    if (element) {
      this.renderer.setProperty(element, 'textContent', timeString);
    }
  }
}
