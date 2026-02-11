# 🎓 技术教程项目开发技能

## 📍 Skill位置

本项目的开发技能已经提取为独立的可复用技能，位于：

```
../tutorial-project-development-skill/tutorial-project-development/
```

**注意**：Skill目录在项目外部，不会被Git跟踪。这样你可以：
- ✅ 将skill用于其他项目
- ✅ 独立更新和维护skill
- ✅ 避免混淆项目内容和开发工具

## 🚀 快速使用

### 方式1：在其他项目中使用

```bash
# 1. 复制skill到新项目
cp -r ../tutorial-project-development-skill/tutorial-project-development my-new-project/skills/

# 2. 使用模板
cp my-new-project/skills/templates/project-plan.md my-new-project/PLAN.md

# 3. 按照技能指导开发教程
# 参考 my-new-project/skills/tutorial-project-development/SKILL.md
```

### 方式2：在Claude Code中调用

```
"使用../tutorial-project-development-skill/中的技能，帮我规划一个Vue.js教程项目"
```

## 📦 Skill内容

### 核心文件

- **SKILL.md** - 完整技能定义（8大能力、工作流程、最佳实践）
- **README.md** - 技能使用指南
- **QUICKSTART.md** - 5分钟快速开始
- **SKILL_SUMMARY.md** - 技能总结

### 模板文件

- **templates/project-plan.md** - 项目规划模板
- **templates/chapter-template.md** - 章节内容模板
- **templates/notebook-template.ipynb** - Jupyter Notebook模板

## 💡 核心能力

1. **项目规划** - 需求分析、大纲设计、学习路径
2. **结构设计** - 目录组织（类型分组vs模块嵌套）
3. **文档编写** - Markdown + 代码示例 + 图表
4. **Notebook创建** - 交互式学习、可执行代码
5. **案例开发** - 端到端项目、难度递进
6. **GitHub管理** - 仓库配置、版本控制
7. **MkDocs部署** - 专业网站、自动化部署
8. **迭代优化** - 问题修复、性能优化

## 🎯 适用场景

- 编程语言教程
- 框架使用指南
- 工具使用手册
- 技术概念讲解
- 任何技术教程项目

## 📚 基于真实经验

本技能完全基于RAG完整教程项目的真实开发经验：

- ✅ 4个模块，20章文档
- ✅ 17个Jupyter Notebooks
- ✅ 6个实战案例（入门到专家）
- ✅ ~500,000字内容
- ✅ MkDocs Material专业部署
- ✅ GitHub Pages在线发布

**项目地址**：https://github.com/vivy-yi/rag-tutorial
**在线文档**：https://vivy-yi.github.io/rag-tutorial/

## 🔄 持续改进

这个技能是开源的，欢迎：
- 根据使用经验改进
- 添加新的模板
- 分享你的使用案例
- 贡献反馈和建议

## 📞 获取帮助

查看skill目录中的详细文档：
- 开始：`QUICKSTART.md`
- 深入：`SKILL.md`
- 参考：`README.md`

---

**版本**: 1.0.0
**创建日期**: 2026-02-11
**基于项目**: RAG完整教程
**维护者**: vivy-yi + Claude Sonnet 4.5
