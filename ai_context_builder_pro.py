"""
AI Context Builder Pro - 專業版
====================================
自動將 Python 專案轉化為高品質 AI Prompt

主要特色：
- 🚀 快取機制 - 避免重複掃描
- 🔥 智能排除 - 自動過濾無用目錄
- 📊 多格式輸出 - XML/JSON/Markdown
- 📈 進度追蹤 - 實時顯示掃描狀態
- 💾 配置管理 - 儲存常用設定
- 🎨 優化界面 - 清晰易用的操作流程
"""

from __future__ import annotations
import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import xml.etree.ElementTree as ET
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 配置常數
# ==========================================
DEFAULT_EXCLUDE_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "env",
    "node_modules", ".pytest_cache", ".mypy_cache",
    "build", "dist", "*.egg-info", ".tox", ".idea", ".vscode"
}

CONFIG_FILE = Path.home() / ".ai_context_builder_config.json"

# ==========================================
# 核心模組 1：AST 專案掃描器
# ==========================================
class ProjectScanner(ast.NodeVisitor):
    """Python 檔案結構掃描器 - 萃取類別、函式、docstring"""
    
    def __init__(self, file_path: Path | str) -> None:
        self.file_path = Path(file_path)
        self.metadata: dict[str, Any] = {
            "path": str(file_path),
            "classes": [],
            "functions": [],
            "imports": [],
            "lines": 0,
            "size_bytes": 0
        }
        self._class_depth = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """訪問類別定義"""
        methods = [
            m.name
            for m in node.body
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        bases = [self._get_node_name(base) for base in node.bases]
        
        self.metadata["classes"].append({
            "name": node.name,
            "doc": ast.get_docstring(node) or "",
            "methods": methods,
            "bases": bases,
            "lineno": node.lineno
        })
        self._class_depth += 1
        self.generic_visit(node)
        self._class_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """訪問函式定義（僅模組層級）"""
        if self._class_depth > 0:
            self.generic_visit(node)
            return

        args = [arg.arg for arg in node.args.args]
        self.metadata["functions"].append({
            "name": node.name,
            "doc": ast.get_docstring(node) or "",
            "args": args,
            "lineno": node.lineno
        })
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """訪問異步函式定義"""
        if self._class_depth > 0:
            self.generic_visit(node)
            return
        
        args = [arg.arg for arg in node.args.args]
        self.metadata["functions"].append({
            "name": f"async {node.name}",
            "doc": ast.get_docstring(node) or "",
            "args": args,
            "lineno": node.lineno
        })
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """訪問 import 語句"""
        for alias in node.names:
            self.metadata["imports"].append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """訪問 from...import 語句"""
        if node.module:
            self.metadata["imports"].append(node.module)
        self.generic_visit(node)

    def _get_node_name(self, node: ast.expr) -> str:
        """獲取節點名稱"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_node_name(node.value)}.{node.attr}"
        return str(node)

    def collect_metadata(self) -> dict[str, Any]:
        """收集檔案基本資訊"""
        try:
            self.metadata["size_bytes"] = self.file_path.stat().st_size
            self.metadata["lines"] = len(self.file_path.read_text(encoding="utf-8").splitlines())
        except Exception:
            pass
        return self.metadata


# ==========================================
# 快取掃描結果
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def cached_scan_project(
    target_path: str, 
    exclude_dirs: tuple[str, ...]
) -> list[dict[str, Any]]:
    """快取版本的專案掃描（5分鐘有效）"""
    return scan_project(Path(target_path), set(exclude_dirs))


def scan_project(
    target_dir: Path,
    exclude_dirs: set[str] = None
) -> list[dict[str, Any]]:
    """
    掃描專案目錄，萃取所有 Python 檔案的結構資訊
    
    Args:
        target_dir: 專案根目錄
        exclude_dirs: 要排除的目錄名稱集合
    
    Returns:
        模組資訊列表
    """
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS

    modules: list[dict[str, Any]] = []
    files = list(target_dir.rglob("*.py"))
    
    # 建立進度條
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file_path in enumerate(files):
        # 檢查是否在排除目錄中
        if any(excluded in file_path.parts for excluded in exclude_dirs):
            continue
        
        # 更新進度
        progress = (idx + 1) / len(files)
        progress_bar.progress(progress)
        status_text.text(f"掃描中... {file_path.name} ({idx + 1}/{len(files)})")
        
        try:
            source = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            continue

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            continue

        scanner = ProjectScanner(file_path)
        scanner.visit(tree)
        modules.append(scanner.collect_metadata())
    
    progress_bar.empty()
    status_text.empty()
    
    return modules


# ==========================================
# 核心模組 2：MMR 檢索與重排
# ==========================================
def apply_mmr(
    modules: list[dict[str, Any]],
    user_query: str,
    top_n: int = 5,
    lambda_param: float = 0.6,
) -> list[dict[str, Any]]:
    """
    實作 MMR 演算法：篩選出與需求最相關且內容不重複的模組
    
    Args:
        modules: 所有模組列表
        user_query: 使用者查詢
        top_n: 返回前 N 個結果
        lambda_param: 相關性 vs 多樣性權重（0-1）
    
    Returns:
        篩選後的模組列表
    """
    if not modules:
        return []

    # 1. 將結構化資料轉為純文字特徵
    texts = [user_query]
    for mod in modules:
        classes_str = " ".join([c["name"] for c in mod.get("classes", [])])
        funcs_str = " ".join([f["name"] for f in mod.get("functions", [])])
        imports_str = " ".join(mod.get("imports", [])[:10])  # 限制 import 數量
        doc_text = f"Path: {mod['path']} Classes: {classes_str} Functions: {funcs_str} Imports: {imports_str}"
        texts.append(doc_text)

    # 2. 計算 TF-IDF 與相似度
    vectorizer = TfidfVectorizer(max_features=500)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return modules[:top_n]
    
    query_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    doc_sim_matrix = cosine_similarity(tfidf_matrix[1:])

    # 3. 執行 MMR 迭代
    unselected = list(range(len(modules)))
    selected_indices = []
    top_n = min(top_n, len(modules))

    while len(selected_indices) < top_n and unselected:
        mmr_scores = {}
        for i in unselected:
            relevance = query_sim[i]
            if not selected_indices:
                penalty = 0.0
            else:
                penalty = max([doc_sim_matrix[i][j] for j in selected_indices])
                
            mmr_score = lambda_param * relevance - (1 - lambda_param) * penalty
            mmr_scores[i] = mmr_score
            
        best_idx = max(mmr_scores, key=mmr_scores.get)
        selected_indices.append(best_idx)
        unselected.remove(best_idx)

    return [modules[i] for i in selected_indices]


# ==========================================
# 核心模組 3：夾攻排序（Bookending）
# ==========================================
def apply_bookending(modules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    實作夾攻策略：將最重要的內容放在開頭和結尾
    順序：[最重要, 輔助資訊..., 次重要]
    
    根據認知心理學的系列位置效應（Serial Position Effect）：
    - 首位效應（Primacy Effect）：開頭資訊記憶最深
    - 近因效應（Recency Effect）：結尾資訊也記得清楚
    """
    if len(modules) < 2: 
        return modules
    
    best = modules[0]
    second_best = modules[1]
    middle = modules[2:]
    
    return [best] + middle + [second_best]


# ==========================================
# 核心模組 4：多格式輸出
# ==========================================
def format_to_xml(sorted_modules: list[dict[str, Any]], query: str) -> str:
    """將模組清單格式化為 XML"""
    root = ET.Element("context")
    root.set("timestamp", datetime.now().isoformat())
    root.set("total_modules", str(len(sorted_modules)))

    for idx, mod in enumerate(sorted_modules, 1):
        doc = ET.SubElement(root, "document", {
            "id": str(mod.get("path", "")),
            "priority": "high" if idx in (1, len(sorted_modules)) else "medium"
        })
        content = ET.SubElement(doc, "content")
        content.text = json.dumps(mod, ensure_ascii=False, indent=2)

    query_node = ET.SubElement(root, "query")
    query_node.text = query

    return ET.tostring(root, encoding="unicode", method="xml")


def format_to_json(sorted_modules: list[dict[str, Any]], query: str) -> str:
    """將模組清單格式化為 JSON"""
    output = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "total_modules": len(sorted_modules),
        "modules": sorted_modules
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


def format_to_markdown(sorted_modules: list[dict[str, Any]], query: str) -> str:
    """將模組清單格式化為 Markdown"""
    lines = [
        f"# AI Context - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"## 任務需求",
        f"> {query}",
        "",
        f"## 相關模組 ({len(sorted_modules)} 個)",
        ""
    ]
    
    for idx, mod in enumerate(sorted_modules, 1):
        priority = "🔥 最重要" if idx == 1 else ("⭐ 次要" if idx == len(sorted_modules) else "📄 輔助")
        lines.append(f"### {idx}. {priority} - {Path(mod['path']).name}")
        lines.append(f"**路徑**: `{mod['path']}`")
        lines.append(f"**大小**: {mod.get('size_bytes', 0):,} bytes | **行數**: {mod.get('lines', 0):,}")
        lines.append("")
        
        if mod.get("classes"):
            lines.append("**類別**:")
            for cls in mod["classes"]:
                methods_str = ", ".join(cls["methods"][:5])
                if len(cls["methods"]) > 5:
                    methods_str += f" ... (+{len(cls['methods']) - 5})"
                lines.append(f"- `{cls['name']}`: {methods_str}")
            lines.append("")
        
        if mod.get("functions"):
            lines.append("**函式**:")
            for func in mod["functions"][:10]:
                lines.append(f"- `{func['name']}`")
            if len(mod["functions"]) > 10:
                lines.append(f"  ... (+{len(mod['functions']) - 10} 個)")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


def build_full_prompt(context: str, query: str, format_type: str) -> str:
    """
    建立完整的 AI Prompt
    
    Args:
        context: 格式化後的上下文資訊
        query: 使用者查詢
        format_type: 格式類型
    
    Returns:
        完整的 Prompt 文字
    """
    if format_type == "markdown":
        return context  # Markdown 已包含完整資訊
    
    return f"""# AI 系統提示詞
## 你的角色
你是一位資深的 Python 系統架構師，擅長分析現有程式碼並實作新功能。

## 任務說明
請根據下方的專案上下文資訊，協助我完成以下需求：

**需求描述**:
{query}

## 夾攻規範（Bookending Strategy）
根據認知心理學的系列位置效應：
1. **第一個模組**是最核心的程式碼，請嚴格遵守其設計模式和架構風格
2. **最後一個模組**是次要但關鍵的依賴，請確保你的實作與它相容
3. **中間的模組**提供輔助參考，可彈性參考

## 專案上下文資訊
{context}

## 輸出要求
- 提供完整可執行的程式碼
- 遵循現有程式碼風格
- 包含適當的錯誤處理和類型提示
- 添加清晰的註解和 docstring
"""


# ==========================================
# 配置管理
# ==========================================
def load_config() -> dict[str, Any]:
    """載入使用者配置"""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {
        "default_path": str(Path.cwd()),
        "top_n": 5,
        "lambda_param": 0.6,
        "output_format": "xml",
        "exclude_dirs": list(DEFAULT_EXCLUDE_DIRS)
    }


def save_config(config: dict[str, Any]) -> None:
    """儲存使用者配置"""
    try:
        CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    except Exception as e:
        st.warning(f"配置儲存失敗: {e}")


# ==========================================
# Streamlit UI 介面
# ==========================================
def setup_page() -> None:
    """設置頁面配置"""
    st.set_page_config(
        page_title="AI Context Builder Pro",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 自訂 CSS
    st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .success-box {
        padding: 20px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar(config: dict[str, Any]) -> dict[str, Any]:
    """渲染側邊欄配置"""
    with st.sidebar:
        st.title("🛠️ 配置選項")
        
        st.subheader("📊 進階參數")
        top_n = st.slider(
            "保留模組數量",
            min_value=1,
            max_value=20,
            value=config.get("top_n", 5),
            help="選擇最相關的前 N 個模組"
        )
        
        lambda_param = st.slider(
            "相關性 vs 多樣性",
            min_value=0.1,
            max_value=0.9,
            value=config.get("lambda_param", 0.6),
            step=0.1,
            help="值越高 = 越注重與任務的相關性\n值越低 = 越注重內容多樣性"
        )
        
        output_format = st.selectbox(
            "輸出格式",
            options=["xml", "json", "markdown"],
            index=["xml", "json", "markdown"].index(config.get("output_format", "xml")),
            help="選擇輸出格式"
        )
        
        st.subheader("🚫 排除目錄")
        exclude_text = st.text_area(
            "要排除的目錄（每行一個）",
            value="\n".join(config.get("exclude_dirs", list(DEFAULT_EXCLUDE_DIRS))),
            height=150,
            help="這些目錄將不會被掃描"
        )
        exclude_dirs = [line.strip() for line in exclude_text.split("\n") if line.strip()]
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 儲存配置", use_container_width=True):
                new_config = {
                    "default_path": config.get("default_path", str(Path.cwd())),
                    "top_n": top_n,
                    "lambda_param": lambda_param,
                    "output_format": output_format,
                    "exclude_dirs": exclude_dirs
                }
                save_config(new_config)
                st.success("配置已儲存！")
        
        with col2:
            if st.button("🔄 重置", use_container_width=True):
                st.rerun()
        
        st.divider()
        st.caption("💡 提示：配置會自動儲存到使用者目錄")
        
        return {
            "top_n": top_n,
            "lambda_param": lambda_param,
            "output_format": output_format,
            "exclude_dirs": exclude_dirs
        }


def render_main_interface(config: dict[str, Any], settings: dict[str, Any]) -> None:
    """渲染主介面"""
    st.title("🏗️ AI Context Builder Pro")
    st.markdown("""
    <div class="success-box">
    <h4>🎯 功能特色</h4>
    <ul>
        <li>🚀 <b>智能掃描</b>：自動分析 Python 專案結構</li>
        <li>🧠 <b>MMR 演算法</b>：智能篩選最相關模組</li>
        <li>📍 <b>夾攻策略</b>：將重要資訊放在開頭和結尾</li>
        <li>📦 <b>多格式輸出</b>：支援 XML / JSON / Markdown</li>
        <li>💾 <b>快取機制</b>：加快重複操作速度</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # 主輸入表單
    with st.form("main_form"):
        st.subheader("📝 輸入資訊")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            target_path = st.text_input(
                "專案路徑",
                value=config.get("default_path", str(Path.cwd())),
                help="輸入要掃描的專案根目錄"
            )
            
            use_cache = st.checkbox(
                "使用快取",
                value=True,
                help="啟用後會快取掃描結果（5分鐘有效）"
            )
        
        with col2:
            user_query = st.text_area(
                "任務需求",
                placeholder="例如：新增一個使用者登入 API，包含 JWT 驗證和錯誤處理...",
                height=120,
                help="描述你希望 AI 執行的任務，越具體越好"
            )
        
        submitted = st.form_submit_button(
            "🔥 產生 AI Context",
            use_container_width=True,
            type="primary"
        )
    
    # 處理表單提交
    if submitted:
        if not user_query.strip():
            st.error("❌ 請輸入任務需求！")
            return
        
        target = Path(target_path.strip())
        if not target.exists():
            st.error(f"❌ 路徑不存在: {target}")
            return
        
        if not target.is_dir():
            st.error(f"❌ 路徑不是目錄: {target}")
            return
        
        # 執行掃描和分析
        process_and_display_results(
            target_path=str(target),
            user_query=user_query.strip(),
            settings=settings,
            use_cache=use_cache
        )


def process_and_display_results(
    target_path: str,
    user_query: str,
    settings: dict[str, Any],
    use_cache: bool
) -> None:
    """處理並顯示結果"""
    
    with st.spinner("⏳ 正在分析專案..."):
        # Step 1: 掃描專案
        if use_cache:
            raw_modules = cached_scan_project(
                target_path,
                tuple(settings["exclude_dirs"])
            )
        else:
            raw_modules = scan_project(
                Path(target_path),
                set(settings["exclude_dirs"])
            )
        
        if not raw_modules:
            st.warning("⚠️ 沒有找到可分析的 Python 檔案")
            return
        
        # Step 2: MMR 篩選
        filtered_modules = apply_mmr(
            raw_modules,
            user_query,
            top_n=settings["top_n"],
            lambda_param=settings["lambda_param"]
        )
        
        # Step 3: Bookending 排序
        final_modules = apply_bookending(filtered_modules)
        
        # Step 4: 格式化輸出
        format_type = settings["output_format"]
        if format_type == "xml":
            context = format_to_xml(final_modules, user_query)
        elif format_type == "json":
            context = format_to_json(final_modules, user_query)
        else:  # markdown
            context = format_to_markdown(final_modules, user_query)
        
        full_prompt = build_full_prompt(context, user_query, format_type)
    
    # 顯示結果
    st.success("✅ 分析完成！")
    
    # 統計資訊
    col1, col2, col3, col4 = st.columns(4)
    
    total_size = sum(m.get("size_bytes", 0) for m in raw_modules)
    total_lines = sum(m.get("lines", 0) for m in raw_modules)
    
    col1.metric("掃描檔案", f"{len(raw_modules)} 個")
    col2.metric("選中模組", f"{len(final_modules)} 個")
    col3.metric("總大小", f"{total_size / 1024:.1f} KB")
    col4.metric("總行數", f"{total_lines:,}")
    
    # 標籤頁顯示結果
    tab1, tab2, tab3 = st.tabs(["📋 完整 Prompt", "🔍 模組詳情", "📊 統計分析"])
    
    with tab1:
        st.subheader("完整 AI Prompt")
        st.code(full_prompt, language="markdown" if format_type == "markdown" else format_type)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label=f"💾 下載 ({format_type.upper()})",
                data=full_prompt,
                file_name=f"ai_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}",
                mime=f"application/{format_type}",
                use_container_width=True
            )
        with col2:
            if st.button("📋 複製到剪貼簿", use_container_width=True):
                st.info("請手動選取並複製上方文字")
    
    with tab2:
        st.subheader("選中的模組詳情")
        for idx, mod in enumerate(final_modules, 1):
            priority_emoji = "🔥" if idx == 1 else ("⭐" if idx == len(final_modules) else "📄")
            priority_text = "最重要" if idx == 1 else ("次要" if idx == len(final_modules) else "輔助")
            
            with st.expander(f"{priority_emoji} {idx}. {Path(mod['path']).name} - {priority_text}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("類別數", len(mod.get("classes", [])))
                col2.metric("函式數", len(mod.get("functions", [])))
                col3.metric("行數", mod.get("lines", 0))
                
                st.code(mod["path"], language="text")
                
                if mod.get("classes"):
                    st.markdown("**類別:**")
                    for cls in mod["classes"]:
                        st.markdown(f"- `{cls['name']}` (第 {cls['lineno']} 行)")
                        if cls["methods"]:
                            st.markdown(f"  方法: {', '.join(cls['methods'][:10])}")
                
                if mod.get("functions"):
                    st.markdown("**函式:**")
                    for func in mod["functions"][:10]:
                        st.markdown(f"- `{func['name']}` (第 {func['lineno']} 行)")
    
    with tab3:
        st.subheader("統計分析")
        
        # 模組大小分布
        st.markdown("### 📏 模組大小分布")
        sizes = [m.get("size_bytes", 0) for m in final_modules]
        names = [Path(m["path"]).name for m in final_modules]
        
        import pandas as pd
        df = pd.DataFrame({
            "模組": names,
            "大小 (KB)": [s / 1024 for s in sizes],
            "行數": [m.get("lines", 0) for m in final_modules],
            "類別數": [len(m.get("classes", [])) for m in final_modules],
            "函式數": [len(m.get("functions", [])) for m in final_modules]
        })
        st.dataframe(df, use_container_width=True)
        
        st.markdown("### 📦 專案概覽")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("平均檔案大小", f"{total_size / len(raw_modules) / 1024:.1f} KB")
            st.metric("平均行數", f"{total_lines / len(raw_modules):.0f}")
        with col2:
            total_classes = sum(len(m.get("classes", [])) for m in raw_modules)
            total_functions = sum(len(m.get("functions", [])) for m in raw_modules)
            st.metric("總類別數", total_classes)
            st.metric("總函式數", total_functions)


# ==========================================
# 主程式入口
# ==========================================
def main():
    """主程式"""
    setup_page()
    
    # 載入配置
    config = load_config()
    
    # 渲染側邊欄並獲取設定
    settings = render_sidebar(config)
    
    # 渲染主介面
    render_main_interface(config, settings)
    
    # 頁尾資訊
    st.divider()
    st.caption("💡 提示：首次掃描較慢，之後會使用快取加速 | Made with ❤️ using Streamlit")


if __name__ == "__main__":
    main()
