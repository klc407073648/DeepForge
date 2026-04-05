import {
  Component,
  CUSTOM_ELEMENTS_SCHEMA,
  OnInit,
  Inject,
  PLATFORM_ID,
} from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { NzDropDownModule } from 'ng-zorro-antd/dropdown';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzIconModule } from 'ng-zorro-antd/icon';
import { NzMenuModule } from 'ng-zorro-antd/menu';
import { TranslateService, TranslateModule } from '@ngx-translate/core';

export type SupportedLanguage = 'zh' | 'en';

@Component({
  selector: 'app-language-switcher',
  standalone: true,
  imports: [
    CommonModule,
    NzDropDownModule,
    NzButtonModule,
    NzIconModule,
    NzMenuModule,
    TranslateModule,
  ],
  templateUrl: './language-switcher.component.html',
  styleUrls: ['./language-switcher.component.css'],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
})
export class LanguageSwitcherComponent implements OnInit {
  currentLanguage: SupportedLanguage = 'zh';
  supportedLanguages: SupportedLanguage[] = ['zh', 'en'];

  private languageDisplayNames: { [key in SupportedLanguage]: string } = {
    zh: '中文',
    en: 'English',
  };

  private languageFlags: { [key in SupportedLanguage]: string } = {
    zh: '🇨🇳',
    en: '🇺🇸',
  };

  constructor(
    private translate: TranslateService,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {}

  ngOnInit(): void {
    // 设置默认语言
    this.translate.setDefaultLang('zh');

    // 从本地存储获取语言设置（仅在浏览器环境中）
    if (isPlatformBrowser(this.platformId)) {
      const savedLanguage = localStorage.getItem(
        'preferredLanguage'
      ) as SupportedLanguage;
      if (savedLanguage && this.supportedLanguages.includes(savedLanguage)) {
        this.currentLanguage = savedLanguage;
        this.translate.use(savedLanguage);
      } else {
        this.currentLanguage = 'zh';
        this.translate.use('zh');
      }
    } else {
      this.translate.use('zh');
    }

    // 监听语言变化
    this.translate.onLangChange.subscribe((event) => {
      this.currentLanguage = event.lang as SupportedLanguage;
    });
  }

  changeLanguage(language: SupportedLanguage): void {
    this.translate.use(language);
    this.currentLanguage = language;

    // 仅在浏览器环境中保存到localStorage
    if (isPlatformBrowser(this.platformId)) {
      localStorage.setItem('preferredLanguage', language);
    }
  }

  getCurrentLanguageFlag(): string {
    return this.languageFlags[this.currentLanguage];
  }

  getCurrentLanguageName(): string {
    return this.languageDisplayNames[this.currentLanguage];
  }

  getLanguageFlag(language: SupportedLanguage): string {
    return this.languageFlags[language];
  }

  getLanguageDisplayName(language: SupportedLanguage): string {
    return this.languageDisplayNames[language];
  }
}
