// if-admin.directive.ts - 管理员权限检查结构型指令
import {
  Directive,
  Input,
  TemplateRef,
  ViewContainerRef,
  OnInit,
} from '@angular/core';

@Directive({
  selector: '[appIfAdmin]',
})
export class IfAdminDirective implements OnInit {
  private hasView = false;

  @Input() appIfAdmin: boolean = false;

  constructor(
    private templateRef: TemplateRef<any>,
    private viewContainer: ViewContainerRef
  ) {}

  ngOnInit() {
    this.updateView();
  }

  private updateView() {
    if (this.appIfAdmin && !this.hasView) {
      this.viewContainer.createEmbeddedView(this.templateRef);
      this.hasView = true;
    } else if (!this.appIfAdmin && this.hasView) {
      this.viewContainer.clear();
      this.hasView = false;
    }
  }
}
