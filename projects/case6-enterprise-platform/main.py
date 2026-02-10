"""
案例6：企业级RAG平台
FastAPI后端主程序
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from rag_engine import EnterpriseRAGEngine
from auth import get_current_user, User
from cache import CacheManager

app = FastAPI(
    title="企业级RAG平台",
    version="1.0.0",
    description="企业级知识库问答系统API"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化
rag_engine = EnterpriseRAGEngine()
cache_manager = CacheManager()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_cache: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    cached: bool = False


@app.get("/")
async def root():
    return {"message": "企业级RAG平台 API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user)
):
    """查询接口（需要认证）"""

    try:
        # 检查缓存
        cache_key = f"{current_user.id}:{request.question}"
        if request.use_cache:
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                return QueryResponse(**cached_result, cached=True)

        # 执行查询
        result = rag_engine.query(
            question=request.question,
            user_id=current_user.id,
            top_k=request.top_k
        )

        response = QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0)
        )

        # 缓存结果
        if request.use_cache:
            cache_manager.set(cache_key, response.dict(), ttl=1800)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """上传文档（需要认证）"""

    try:
        # 保存文件
        content = await file.read()

        # 添加到知识库
        doc_id = rag_engine.add_document(
            content=content.decode(),
            filename=file.filename,
            user_id=current_user.id
        )

        return {"message": "文档上传成功", "document_id": doc_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents(current_user: User = Depends(get_current_user)):
    """列出用户文档"""

    try:
        docs = rag_engine.list_user_documents(current_user.id)
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats(current_user: User = Depends(get_current_user)):
    """获取统计信息"""

    try:
        stats = rag_engine.get_stats(current_user.id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
