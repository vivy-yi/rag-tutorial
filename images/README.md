# RAG教程图片资源

本目录包含教程中使用的所有图片资源，按模块分类组织。

## 目录结构

```
images/
├── module1-basic/         # 模块1：基础入门 (15张)
├── module2-optimization/  # 模块2：核心优化 (20张)
├── module3-advanced/      # 模块3：高级架构 (30张)
├── module4-production/    # 模块4：生产部署 (15张)
└── logos/                 # Logo和品牌资源 (5张)
```

## 获取图片的方法

这些图片来自以下5个开源仓库，你可以通过以下方式获取：

### 方法1：手动下载（推荐）

#### 1. AgenticRAG-Survey (40+张)

**仓库地址**: https://github.com/AutonLab/AgenticRAG-Survey

```bash
# 克隆仓库
git clone https://github.com/AutonLab/AgenticRAG-Survey.git temp_agentic

# 复制图片到对应目录
cp temp_agentic/images/*.png images/module3-advanced/
cp temp_agentic/images/*.svg images/module3-advanced/

# 清理临时文件
rm -rf temp_agentic
```

**主要图片**:
- `overview_agentic_rag.svg`
- `single_agentic_RAG.png`
- `multiagent_agentic_rag.png`
- `agentic_self_reflection.png`
- `agentic_planning.png`
- `graph_rag.png`

#### 2. advanced-rag (20+张)

**仓库地址**: https://github.com/langchain-ai/rag-from-scratch

```bash
# 克隆仓库
git clone https://github.com/langchain-ai/rag-from-scratch.git temp_advanced

# 复制图片
cp temp_advanced/images/*.png images/module1-basic/
cp temp_advanced/images/*.gif images/module1-basic/

# 清理
rm -rf temp_advanced
```

**主要图片**:
- `Advanced_RAG.png`
- `Hybrid_Search.png`
- `Recall_Precision_in_RAG_Diagram.png`
- `advanced-rag-setup.gif`

#### 3. RAG_Techniques (20+张)

**仓库地址**: https://github.com/NirDiamant/RAG_Techniques

```bash
# 克隆仓库
git clone https://github.com/NirDiamant/RAG_Techniques.git temp_techniques

# 复制图片
cp temp_techniques/images/*.svg images/
cp temp_techniques/images/*.png images/

# 清理
rm -rf temp_techniques
```

**主要图片**:
- `fusion_retrieval.svg`
- `reranking_comparison.svg`
- `HyDe.svg`
- `grouse.svg`

---

### 方法2：使用下载脚本（自动化）

我为你创建了一个自动化下载脚本，可以一键获取所有图片。

**脚本位置**: `scripts/download_images.sh`

**使用方法**:

```bash
# 赋予执行权限
chmod +x scripts/download_images.sh

# 运行脚本
./scripts/download_images.sh
```

---

### 方法3：使用占位符（快速开始）

如果你只是想快速查看教程效果，可以先使用占位符图片。

```bash
# 运行占位符生成脚本
python scripts/generate_placeholders.py
```

这会为所有缺失的图片生成简单的占位符图片。

---

## 图片命名规范

- 使用描述性文件名
- 使用连字符分隔单词
- 小写字母
- 例如：`agentic-rag-architecture.svg`

## 在Markdown中引用图片

### 标准格式

```markdown
![图片说明](../images/module3-advanced/agentic_rag.png)

*图13-1：Agentic RAG架构图*
```

### 相对路径说明

- 如果在模块根目录的Markdown文件中引用：
  ```markdown
  ![图片](images/module3-advanced/xxx.png)
  ```

- 如果在子目录（如notebooks）中引用：
  ```markdown
  ![图片](../images/module3-advanced/xxx.png)
  ```

---

## 图片使用清单

### 模块1：基础入门 (15张)

详见：`../图表资源清单.md` 第15-62行

主要图片：
- `naive_rag.png` - 传统Naive RAG架构
- `advanced_rag.png` - Advanced RAG架构
- `Advanced_RAG.png` - 完整流程图
- `Recall_Precision_in_RAG_Diagram.png` - 评估指标图

### 模块2：核心优化 (20张)

详见：`../图表资源清单.md` 第64-113行

主要图片：
- `fusion_retrieval.svg` - 融合检索架构
- `Hybrid_Search.png` - 混合搜索示意图
- `reranking_comparison.svg` - 重排序效果对比
- `contextual_chunk_headers.svg` - 分块策略图

### 模块3：高级架构 (30张)

详见：`../图表资源清单.md` 第115-199行

主要图片：
- `overview_agentic_rag.svg` - Agentic RAG总览
- `agentic_self_reflection.png` - 反思模式
- `graph_rag.png` - Graph RAG架构
- `multi_agent_pattern.png` - 多智能体协作

### 模块4：生产部署 (15张)

详见：`../图表资源清单.md` 第201-233行

主要图片：
- `Advanced_RAG.png` - 生产架构示例
- `zilliz_interface.png` - 云服务界面
- `Recall_Precision_in_RAG_Diagram.png` - 性能监控图

---

## 常见问题

### Q1: 图片显示不出来怎么办？

检查图片路径是否正确，确保使用相对路径。

### Q2: 可以自己制作图片吗？

可以！参考 `../图表资源清单.md` 第333-355行的工具推荐。

### Q3: SVG图片如何编辑？

推荐使用 Inkscape（免费）或 Adobe Illustrator。

### Q4: 图片太大怎么办？

可以使用以下命令批量压缩：

```bash
# 使用ImageMagick
mogrify -resize 800x600 -quality 85 images/*/*.png
```

---

## 维护说明

- 当原仓库更新图片时，需要同步更新
- 建议定期检查图片链接是否有效
- 新增图片时请更新此README

---

**更新日期**: 2025-02-10
**维护者**: RAG教程项目组
