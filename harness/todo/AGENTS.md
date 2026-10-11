项目概述
技术栈
项目结构
常用命令
编码规范
测试规范
修改规则（更改代码前、更改代码后）
Agent 使用约定

# AGENTS.md  
  
## 1. Project Overview  
  
本项目是 XXX 系统，主要负责 XXX 能力。  
  
请在修改代码前先理解项目结构、相关模块和现有测试。  
  
---
  
## 2. Tech Stack  
  
- Language:  
- Framework:  
- Database:  
- Test Framework:  
- Package Manager:  
- CI/CD:  
  
---
  
## 3. Repository Structure  
  
```text  
src/        Source code  
tests/      Test code  
docs/       Documentation  
scripts/    Development and CI scripts  
```  
  
---
  
## 4. Common Commands  
  
```bash  
# Install dependencies  
# TODO  
  
# Run tests  
# TODO  
  
# Run lint  
# TODO  
  
# Build  
# TODO  
```  
  
---
  
## 5. Coding Guidelines  
  
- 遵循现有代码风格  
- 不做无关格式化  
- 不修改无关文件  
- 未经确认不要引入新依赖  
- 新增业务逻辑必须补充测试  
- Bug 修复必须补充回归测试  
- 不要随意修改公共 API  
- 不要绕过已有抽象和模块边界  
  
---
  
## 6. Testing Guidelines  
  
- 新功能必须有单元测试或集成测试  
- 缺陷修复必须有回归测试  
- 重构前优先补充保护性测试  
- 修改完成后应运行相关测试  
- 如果无法运行测试，需要说明原因和风险  
  
---
  
## 7. Change Workflow  
  
### Before Changing Code  
  
1. 阅读相关文件  
2. 理解需求和影响范围  
3. 给出简要实现计划  
4. 明确需要修改的文件  
5. 等待确认，除非用户明确要求直接修改  
  
### While Changing Code  
  
1. 保持修改范围最小  
2. 不做无关重构  
3. 不引入无关依赖  
4. 保持现有风格一致  
5. 优先小步提交式修改  
  
### After Changing Code  
  
1. 说明修改了哪些文件  
2. 说明如何验证  
3. 给出测试结果  
4. 列出潜在风险  
5. 给出后续建议  
  
---
  
## 8. Security Rules  
  
- 禁止写入密码、Token、私钥、证书  
- 禁止在日志中打印敏感信息  
- 禁止提交生产数据库连接串  
- 用户隐私数据必须脱敏  
- 涉及鉴权、越权、支付、数据删除的改动必须额外审查  
- 不要关闭安全检查、测试或 CI 门禁  
  
---
  
## 9. Agent Collaboration Rules  
  
如果工具支持 Agent，请按以下规则选择：  
  
- 架构设计：使用 `architect`  
- 测试设计：使用 `tdd-test-writer`  
- 代码审查：使用 `code-reviewer`  
- 安全审计：使用 `security-auditor`  
- 重构任务：使用 `refactor-agent`  
- 文档生成：使用 `doc-writer`  
  
如果工具支持 Skill，请优先使用：  
  
- TDD/ATDD：`tdd-workflow`  
- 代码审查：`code-review-checklist`  
- 安全审计：`security-checklist`  
- API 设计：`api-design`  
- 重构：`refactoring-pattern`  
  
---
  
## 10. Pull Request Guidelines  
  
提交 PR/MR 时应包含：  
  
1. 变更背景  
2. 修改内容  
3. 测试结果  
4. 风险说明  
5. 回滚方案，若适用  
  
---
  
## 11. Do Not  
  
- 不要修改无关文件  
- 不要大规模格式化代码  
- 不要删除测试  
- 不要绕过权限校验  
- 不要引入未批准依赖  
- 不要修改 CI/CD 配置，除非任务明确要求  
- 不要修改生产配置，除非经过确认    总结归纳