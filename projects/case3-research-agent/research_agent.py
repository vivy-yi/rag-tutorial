"""
案例3：ReAct研究Agent
"""

from typing import List, Dict
from tools import SearchTool, ArxivTool, WikipediaTool


class ResearchAgent:
    """AI研究助手 - ReAct Agent"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.tools = {
            "search": SearchTool(),
            "arxiv": ArxivTool(),
            "wikipedia": WikipediaTool()
        }

    def research(
        self,
        topic: str,
        max_papers: int = 5,
        use_arxiv: bool = True,
        use_wikipedia: bool = True
    ) -> Dict:
        """执行研究任务"""

        sources = []
        summary_parts = []

        # Step 1: 维基百科背景
        if use_wikipedia:
            st.info("📖 正在查询维基百科...")
            wiki_result = self.tools["wikipedia"].search(topic)
            if wiki_result:
                sources.append(wiki_result)
                summary_parts.append(f"**背景知识**：\n{wiki_result['summary'][:500]}...")

        # Step 2: ArXiv论文搜索
        if use_arxiv:
            st.info("📚 正在搜索ArXiv论文...")
            arxiv_results = self.tools["arxiv"].search(topic, max_results=max_papers)
            sources.extend(arxiv_results)

            if arxiv_results:
                summary_parts.append(f"\n**相关研究**：\n找到了{len(arxiv_results)}篇相关论文。")
                for i, paper in enumerate(arxiv_results[:3], 1):
                    summary_parts.append(f"\n{i}. {paper['title']}")
                    summary_parts.append(f"   {paper['abstract'][:200]}...")

        # Step 3: 补充搜索
        st.info("🔍 正在补充搜索...")
        search_result = self.tools["search"].search(f"{topic} tutorial review")
        if search_result:
            summary_parts.append(f"\n**补充资料**：\n{search_result['summary'][:300]}...")

        # 生成报告
        report = self._generate_report(topic, summary_parts)

        return {
            "topic": topic,
            "report": report,
            "sources": sources
        }

    def _generate_report(self, topic: str, parts: List[str]) -> str:
        """生成研究报告"""

        report = f"# {topic} - 研究报告\n\n"
        report += "\n".join(parts)
        report += "\n\n---\n\n"
        report += "**说明**：本报告由AI研究助手自动生成，内容仅供参考。"

        return report
