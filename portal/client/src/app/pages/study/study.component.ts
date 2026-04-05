import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { NzIconModule } from 'ng-zorro-antd/icon';

@Component({
  selector: 'app-study',
  standalone: true,
  imports: [RouterLink, NzIconModule],
  templateUrl: './study.component.html',
  styleUrl: './study.component.css',
})
export class StudyComponent {
  constructor() {}
}
