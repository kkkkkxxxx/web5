import os
from typing import Dict, List, Any
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
import requests
from duckduckgo_search import DDGS
import numpy as np
import re
from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI

class FactChecker:
    def __init__(self, api_base: str, model: str, temperature: float, max_tokens: int):
        # 文心API配置
        self.api_key = "5862f79af437ff09b44c6da222a7a1fd1a0eed52"  # 你的访问令牌
        self.api_base = "https://aistudio.baidu.com/llm/lmapi/v3"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )

        # 初始化嵌入模型
        try:
            self.embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        except Exception as e:
            st.error(f"加载BGE-M3模型错误: {str(e)}")
            self.embedding_model = None

        # 初始化Chroma本地知识库
        try:
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.get_or_create_collection(
                name="my-knowledge-base",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
            )
        except Exception as e:
            st.error(f"加载Chroma错误: {str(e)}")
            self.collection = None

    def extract_claim(self, text: str) -> str:
        system_prompt = "你是一个精确的声明提取助手。分析提供的新闻并总结其中心思想。将中心思想格式化为一个值得核查的陈述，即可以独立验证的声明并以中文回答。输出格式:\nclaim: <claim>"
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=1000
            )
            claims_text = response.choices[0].message.content
            claims = re.findall(r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)', claims_text, re.DOTALL)
            claims = [claim.strip() for claim in claims if claim.strip()]
            if not claims and claims_text.strip():
                claims = [line.strip() for line in claims_text.strip().split('\n') if line.strip()]
            return claims[0] if claims else text
        except Exception as e:
            st.error(f"提取声明错误: {str(e)}")
            return text

    def extract_keyinformation(self, text: str) -> str:
        system_prompt = (
            "你是一个精确的声明提取助手。请你对用户输入执行以下多层级分析：\n"
            "1.识别文本中涉及的所有实体（人物/组织/地点/时间）及其类型\n"
            "2.确定核心事件类型（政治/经济/社会/科技等类别）\n"
            "3.提取关键特征（来源可靠性/证据强度/时效性）\n"
            "4.总结中心思想并转化为可验证的声明\n\n"
            "输出格式要求：\n"
            "claim: <可验证的完整陈述>\n"
            "entities: <实体类型1:实体名称1, 实体类型2:实体名称2...>\n"
            "event_type: <事件主类别>\n"
            "key_features: <特征1:值1, 特征2:值2...>\n\n"
            "注意：请严格按照格式输出，不要添加额外解释"
        )
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=1000
            )
            raw_output = response.choices[0].message.content
            claim_match = re.search(r'claim:\s*(.*?)(?=\n\S+:|\Z)', raw_output, re.DOTALL)
            entities_match = re.search(r'entities:\s*(.*?)(?=\n\S+:|\Z)', raw_output, re.DOTALL)
            event_type_match = re.search(r'event_type:\s*(.*?)(?=\n\S+:|\Z)', raw_output, re.DOTALL)
            key_features_match = re.search(r'key_features:\s*(.*?)(?=\n\S+:|\Z)', raw_output, re.DOTALL)
            claim = claim_match.group(1).strip() if claim_match else "未提取到有效声明"
            entities = entities_match.group(1).strip() if entities_match else "无实体信息"
            event_type = event_type_match.group(1).strip() if event_type_match else "未知事件类型"
            key_features = key_features_match.group(1).strip() if key_features_match else "无特征信息"
            formatted_output = (
                f"claim: {claim}\n"
                f"entities: {entities}\n"
                f"event_type: {event_type}\n"
                f"key_features: {key_features}"
            )
            return formatted_output
        except Exception as e:
            st.error(f"提取关键信息错误: {str(e)}")
            return text

    def search_local_knowledge(self, claim: str, top_k: int = 5) -> List[Dict[str, str]]:
        if not self.collection:
            return []
        try:
            results = self.collection.query(
                query_texts=[claim],
                n_results=top_k
            )
            evidence_docs = []
            for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
                evidence_docs.append({
                    'title': metadata.get('title', ''),
                    'url': metadata.get('url', 'Local Knowledge Base'),
                    'snippet': text
                })
            return evidence_docs
        except Exception as e:
            st.error(f"本地知识库检索错误: {str(e)}")
            return []

    def search_evidence(self, claim: str, num_results: int = 5) -> List[Dict[str, str]]:
        try:
            ddgs = DDGS(timeout=60)
            results = list(ddgs.text(claim, max_results=num_results))
            external_evidence = []
            for result in results:
                external_evidence.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', '')
                })
            local_evidence = self.search_local_knowledge(claim, top_k=num_results)
            combined_evidence = external_evidence + local_evidence
            return combined_evidence
        except Exception as e:
            st.error(f"外部证据检索错误: {str(e)}")
            return []

    def get_evidence_chunks(self, evidence_docs: List[Dict[str, str]], claim: str, chunk_size: int = 200, chunk_overlap: int = 50, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.embedding_model:
            return [{
                'text': "Evidence ranking unavailable - BGE-M3 model could not be loaded.",
                'source': "System",
                'similarity': 0.0
            }]
        if not evidence_docs:
            return [{
                'text': "没有找到相关证据。",
                'source': "系统",
                'similarity': 0.0
            }]
        try:
            all_chunks = []
            for doc in evidence_docs:
                all_chunks.append({
                    'text': doc['title'],
                    'source': doc['url'],
                })
                snippet = doc['snippet']
                if len(snippet) <= chunk_size:
                    all_chunks.append({
                        'text': snippet,
                        'source': doc['url'],
                    })
                else:
                    for i in range(0, len(snippet), chunk_size - chunk_overlap):
                        chunk_text = snippet[i:i + chunk_size]
                        if len(chunk_text) >= chunk_size // 2:
                            all_chunks.append({
                                'text': chunk_text,
                                'source': doc['url'],
                            })
            claim_embedding = self.embedding_model.encode(claim)['dense_vecs']
            chunk_texts = [chunk['text'] for chunk in all_chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts)['dense_vecs']
            similarities = []
            for i, chunk_embedding in enumerate(chunk_embeddings):
                similarity = np.dot(claim_embedding, chunk_embedding) / (
                    np.linalg.norm(claim_embedding) * np.linalg.norm(chunk_embedding)
                )
                similarities.append(float(similarity))
            for i, similarity in enumerate(similarities):
                all_chunks[i]['similarity'] = similarity
            ranked_chunks = sorted(all_chunks, key=lambda x: x['similarity'], reverse=True)
            return ranked_chunks[:top_k]
        except Exception as e:
            st.error(f"证据相关性分析错误: {str(e)}")
            return [{
                'text': f"证据相关性分析错误: {str(e)}",
                'source': "System",
                'similarity': 0.0
            }]

    def evaluate_claim(self, keyinformation: str, claim: str, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, str]:
        system_prompt = (
            "你是一个严格的新闻核查助手，请严格按照如下分析流程：\n"
            "1. 来源可信度核查（权重40%）\n"
            "2. 证据质量评估（权重40%）\n"
            "3. 逻辑完整性审查（权重20%）\n"
            "4. 二元判定标准：TRUE（总权重≥70%且无核心矛盾），FALSE（存在反证链/多处逻辑漏洞或大部分来源为个人博客/匿名平台），PARTIALLY TRUE（部分证据支持但有疑点）\n"
            "输出格式：\n"
            "VERDICT: [TRUE|FALSE|PARTIALLY TRUE]\n"
            "REASONING: \n"
            "1. 来源评级\n"
            "2. 关键证据\n"
            "3. 逻辑缺陷\n"
            "4. 时效性验证\n"
            "请严格按照格式输出，不要添加额外解释。"
        )
        evidence_text = "\n\n".join([
            f"evidence {i+1} (relevance: {chunk['similarity']:.2f}):\n{chunk['text']}\nsource: {chunk['source']}"
            for i, chunk in enumerate(evidence_chunks)
        ])
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"keyinformation: {keyinformation}\n\ntext:{claim}\n\nevidence:\n{evidence_text}"}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            result_text = response.choices[0].message.content
            verdict_match = re.search(r'\s*(TRUE|FALSE|PARTIALLY TRUE).*?$', result_text, re.IGNORECASE | re.MULTILINE)
            verdict = verdict_match.group(1) if verdict_match else "UNVERIFIABLE"
            reasoning_match = re.search(r'REASONING:\s*(.*)', result_text, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else result_text
            return {
                "verdict": verdict,
                "reasoning": reasoning
            }
        except Exception as e:
            st.error(f"评估声明错误: {str(e)}")
            return {
                "verdict": "错误",
                "reasoning": f"评估过程中发生错误: {str(e)}"
            }

    def check_fact(self, text: str) -> Dict[str, Any]:
        """
        Main function to check the factuality of a statement.
        
        Args:
            text: The statement to fact-check
            
        Returns:
            Dictionary with all results of the fact-checking process
        """
        # 1. Extract core claim
        claim = self.extract_claim(text)
        
        result = {
            "original_text": text,
            "claim": claim,
            "results": []
        }
        
        # 搜集关键信息
        key=self.extract_keyinformation(text)
        
        # 2. Search for evidence
        evidence_docs = self.search_evidence(claim)
        
        # 3. Get relevant evidence chunks
        evidence_chunks = self.get_evidence_chunks(evidence_docs, claim)
        
        # 4. Evaluate claim based on evidence
        evaluation = self.evaluate_claim(key, claim, evidence_chunks)
        
        # Add results for this claim
        result={
            "claim": claim,
            "evidence_docs": evidence_docs,
            "evidence_chunks": evidence_chunks,
            "verdict": evaluation["verdict"],
            "reasoning": evaluation["reasoning"]
        }
        
        return result


# Function to be imported in the main Streamlit app
def check_fact(claim: str, api_base: str, model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    """
    Public interface for fact checking to be used by the Streamlit app.
    
    Args:
        claim: The statement to fact-check
        api_base: The base URL for the LLM API
        model: The model to use for fact checking
        temperature: Temperature parameter for LLM
        max_tokens: Maximum tokens for LLM response
        
    Returns:
        Dictionary with all results of the fact-checking process
    """
    checker = FactChecker(api_base, model, temperature, max_tokens)
    return checker.check_fact(claim)