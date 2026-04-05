// highlight.directive.ts
import {
  Directive,
  ElementRef,
  HostListener,
  Input,
  Renderer2,
  OnInit,
} from '@angular/core';

@Directive({
  selector: '[appHighlight]', // 使用方括号表示这是一个属性选择器
})
export class HighlightDirective implements OnInit {
  @Input() appHighlight: string = '#ffeb3b'; // 更明显的黄色
  @Input() defaultColor: string = '#f5f5f5'; // 更明显的默认色

  constructor(
    private el: ElementRef,
    private renderer: Renderer2
  ) {}

  ngOnInit() {
    // 初始化时设置默认背景色
    this.highlight(this.defaultColor);
  }

  @HostListener('mouseenter') onMouseEnter() {
    this.highlight(this.appHighlight);
  }

  @HostListener('mouseleave') onMouseLeave() {
    this.highlight(this.defaultColor);
  }

  private highlight(color: string) {
    this.renderer.setStyle(this.el.nativeElement, 'background-color', color);
    this.renderer.setStyle(this.el.nativeElement, 'transition', 'background-color 0.3s ease');
  }
}
