from typing import List, Optional, Union
from qwen_agent.tools.base import BaseTool, register_tool
from transformers import AutoTokenizer
from tongyi_utils.generation_assisted import GenerationAssistedRetriever


@register_tool("search", allow_overwrite=True)
class SearchToolHandler(BaseTool):
    name = "search"

    def __init__(
        self,
        searcher,
        snippet_max_tokens: int = 512,
        k: int = 5,
        generation_assistant: Optional[GenerationAssistedRetriever] = None,
        generated_doc_position: str = "append",
        generation_context_queries: int = 6,
        generation_context_observations: int = 3,
        generation_context_observation_chars: int = 500,
    ):
        super().__init__()

        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.generation_assistant = generation_assistant
        self.generated_doc_position = generated_doc_position
        self.generation_context_queries = generation_context_queries
        self.generation_context_observations = generation_context_observations
        self.generation_context_observation_chars = generation_context_observation_chars
        self.current_question: Optional[str] = None
        self.search_history: List[str] = []
        self.observation_history: List[str] = []

        self.description = f"Performs a search on a knowledge source: supply a single 'query' string; the tool retrieves the top {self.k} most relevant results."

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def _truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not self.tokenizer:
            return text

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text

    def set_query_context(self, question: str) -> None:
        self.current_question = (question or "").strip()
        self.search_history = []
        self.observation_history = []

    def _merge_generation_context(self, query: str, context: Optional[dict] = None) -> dict:
        external = context or {}
        merged = {
            "step_index": len(self.search_history) + 1,
            "previous_subqueries": self.search_history[-self.generation_context_queries :],
            "previous_observations": self.observation_history[
                -self.generation_context_observations :
            ],
        }
        latest_reasoning = (external.get("latest_reasoning") or "").strip()
        if latest_reasoning:
            merged["latest_reasoning"] = latest_reasoning[
                : self.generation_context_observation_chars
            ]
        return merged

    def _build_observation_summary(
        self, query: str, formatted_results: List[str], generated_doc: Optional[str]
    ) -> str:
        parts = [f"Query: {query}"]
        if formatted_results:
            top_results = "\n\n".join(formatted_results[:2])
            parts.append(
                "Top retrieved snippets:\n"
                + top_results[: self.generation_context_observation_chars]
            )
        else:
            parts.append("Top retrieved snippets: [none]")

        if generated_doc:
            parts.append(
                "Generated evidence summary:\n"
                + generated_doc[: self.generation_context_observation_chars]
            )
        return "\n".join(parts)

    def _format_results(self, results: List[dict], max_tokens: int):
        formatted = []
        for idx, r in enumerate(results, 1):
            passage_text = r["text"]
            title = ""

            if passage_text.startswith("---\ntitle:"):
                lines = passage_text.split("\n")
                if len(lines) > 1:
                    title = lines[1].replace("title:", "").strip().strip("\"")

            if not title:
                first_line = passage_text.split('\n')[0].strip()
                title = first_line[:50] + "..." if len(first_line) > 50 else first_line

            snippet = self._truncate(passage_text, max_tokens)

            # Follows original tongyi search tool format
            formatted_result = f"{idx}. <docid: {r['docid']}> [{title}]\n{snippet}"
            formatted.append(formatted_result)

        return formatted

    def search_with_searcher(
        self, query: str, k: int = None, generation_context: Optional[dict] = None
    ):
        try:
            if k is None:
                k = self.k

            results = self.searcher.search(query, k)
            generated_doc = None
            if (
                self.generation_assistant is not None
                and self.current_question
                and isinstance(query, str)
                and query.strip()
            ):
                full_generation_context = self._merge_generation_context(
                    query, context=generation_context
                )
                generated_doc = self.generation_assistant.generate_document(
                    self.current_question, query, context=full_generation_context
                )

            if not results and not generated_doc:
                return f"No results found for '{query}'. Try with a more general query.", []

            docids = []
            for r in results:
                if "docid" in r:
                    docids.append(str(r["docid"]))

            formatted_results = self._format_results(results, self.snippet_max_tokens)
            web_section = "## Web Results\n"
            if formatted_results:
                web_section += "\n\n".join(formatted_results)
            else:
                web_section += "[No corpus results returned.]"

            generated_section = ""
            if generated_doc:
                generated_snippet = self._truncate(generated_doc, self.snippet_max_tokens)
                generated_section = (
                    "## Generated Evidence (Parametric)\n"
                    "1. <docid: generated:parametric> [Model-generated evidence]\n"
                    f"{generated_snippet}"
                )

            sections = []
            if self.generated_doc_position == "prepend" and generated_section:
                sections.append(generated_section)
            sections.append(web_section)
            if self.generated_doc_position != "prepend" and generated_section:
                sections.append(generated_section)

            content = (
                f"A search for '{query}' found {len(formatted_results)} retrieved results."
                "\n\n"
                + "\n\n".join(sections)
            )
            self.search_history.append(query)
            self.observation_history.append(
                self._build_observation_summary(query, formatted_results, generated_doc)
            )
            return content, docids

        except Exception as e:
            return f"Search error for query '{query}': {str(e)}", []

    def call(self, params: Union[str, dict], **kwargs):
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field", None

        if not isinstance(query, str):
            if isinstance(query, list) and len(query) == 1:
                query = query[0]
            else:
                return "[Search] Invalid request format: 'query' must be a string, not an array", None

        generation_context = kwargs.get("generation_context") or {}
        response, docids = self.search_with_searcher(
            query, generation_context=generation_context
        )
        return response, docids
