"""
Batch Feature Extractability Analyzer for GitHub Repositories
Reads repos from CSV, analyzes them, and generates a report
"""

import ast
import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

try:
    from community import community_louvain

    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print(
        "Warning: python-louvain not installed. Modularity calculations will be disabled."
    )


@dataclass
class ExtractabilityMetrics:
    """Metrics for measuring feature extractability"""

    repo_name: str
    repo_url: str
    total_files: int
    total_python_files: int
    internal_dependencies: int
    external_dependencies: int
    fundamental_ml_dependencies: int
    cross_module_calls: int
    coupling_score: float  # 0-1, lower is better
    cohesion_score: float  # 0-1, higher is better
    modularity_score: float  # NEW: -1 to 1, higher is better
    complexity_score: float
    dependency_depth: int
    max_inheritance_depth: int  # NEW
    registry_pattern_usage: int  # NEW
    has_config_system: bool  # NEW
    extractability_score: float  # 0-100, higher means easier to extract
    avg_file_complexity: float
    max_file_complexity: float
    total_classes: int
    total_functions: int
    analysis_status: str  # 'success', 'error', 'no_python_files'
    error_message: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def get_csv_headers(cls):
        """Get CSV header names"""
        return [f.name for f in fields(cls)]


class DependencyAnalyzer:
    """Analyzes dependencies within Python code"""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.module_graph = nx.DiGraph()
        self.imports_map = defaultdict(set)
        self.function_calls = defaultdict(set)
        self.class_definitions = {}
        self.file_complexities = {}
        self.file_analyses = {}
        self.base_packages = set()

    def analyze_file(self, filepath: str) -> Dict:
        """Analyze a single Python file"""
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if not content.strip():
                    return {}
                tree = ast.parse(content, filename=filepath)
        except (SyntaxError, UnicodeDecodeError, Exception) as e:
            return {}

        relative_path = os.path.relpath(filepath, self.repo_path)

        analysis = {
            "imports": set(),
            "function_calls": set(),
            "classes": [],
            "functions": [],
            "complexity": 0,
            "decorators": [],
        }

        for node in ast.walk(tree):
            # Track imports
            if isinstance(node, ast.Import):
                for name in node.names:
                    analysis["imports"].add(name.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    analysis["imports"].add(node.module)

            # Track class definitions with inheritance depth
            elif isinstance(node, ast.ClassDef):
                # Track decorators for registry pattern detection
                decorators = [self._get_name(d) for d in node.decorator_list]
                analysis["decorators"].extend(decorators)

                analysis["classes"].append(
                    {
                        "name": node.name,
                        "methods": [
                            m.name for m in node.body if isinstance(m, ast.FunctionDef)
                        ],
                        "bases": [self._get_name(base) for base in node.bases],
                        "decorators": decorators,
                    }
                )

            # Track function definitions
            elif isinstance(node, ast.FunctionDef):
                analysis["functions"].append(node.name)
                # Track decorators
                decorators = [self._get_name(d) for d in node.decorator_list]
                analysis["decorators"].extend(decorators)
                # Measure complexity (simplified cyclomatic)
                analysis["complexity"] += self._calculate_complexity(node)

            # Track function calls
            elif isinstance(node, ast.Call):
                call_name = self._get_name(node.func)
                if call_name:
                    analysis["function_calls"].add(call_name)

        self.imports_map[relative_path] = analysis["imports"]
        self.function_calls[relative_path] = analysis["function_calls"]
        self.file_complexities[relative_path] = analysis["complexity"]
        self.file_analyses[relative_path] = analysis

        return analysis

    def _get_name(self, node) -> str:
        """Extract name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value_name = self._get_name(node.value)
            if value_name:
                return f"{value_name}.{node.attr}"
            return node.attr
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        return ""

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def find_all_python_files(self) -> List[str]:
        """Find all Python files in the repository"""
        python_files = []
        for root, dirs, files in os.walk(self.repo_path):
            # Skip common non-source directories
            dirs[:] = [
                d
                for d in dirs
                if d
                not in [
                    ".git",
                    "__pycache__",
                    "venv",
                    "env",
                    ".venv",
                    "node_modules",
                    ".tox",
                    "build",
                    "dist",
                    ".eggs",
                ]
            ]

            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    python_files.append(full_path)

        return python_files

    def _detect_base_packages(self, feature_files: List[str]) -> Set[str]:
        """Detect the main package names in the repo"""
        packages = set()

        # Look for common Python package indicators
        for file in feature_files:
            rel_path = Path(file).relative_to(self.repo_path)
            parts = rel_path.parts

            # Skip if it's in root or starts with dot/underscore
            if len(parts) <= 1 or parts[0].startswith(".") or parts[0].startswith("_"):
                continue

            # Check if this directory has __init__.py (making it a package)
            potential_package_dir = Path(self.repo_path) / parts[0]
            if potential_package_dir.is_dir():
                init_file = potential_package_dir / "__init__.py"
                if init_file.exists():
                    packages.add(parts[0])

        # If no packages found through __init__.py, use heuristic
        if not packages:
            dir_counts = defaultdict(int)
            for file in feature_files:
                rel_path = Path(file).relative_to(self.repo_path)
                if len(rel_path.parts) > 1:
                    dir_counts[rel_path.parts[0]] += 1

            # Take directories with most Python files
            if dir_counts:
                max_count = max(dir_counts.values())
                packages = {d for d, c in dir_counts.items() if c >= max_count * 0.3}

        return packages

    def build_dependency_graph(self, feature_files: List[str]):
        """Build directed graph of dependencies"""
        # First, detect base packages
        self.base_packages = self._detect_base_packages(feature_files)

        # Analyze all files
        for filepath in feature_files:
            if os.path.exists(filepath) and filepath.endswith(".py"):
                self.analyze_file(filepath)

        # Build graph edges
        for file, imports in self.imports_map.items():
            for imp in imports:
                self.module_graph.add_edge(file, imp)

    def get_internal_vs_external_deps(
        self, feature_files: List[str]
    ) -> Tuple[Set, Set, Set]:
        """Separate internal (within repo) from external dependencies"""
        internal_deps = set()
        external_deps = set()
        fundamental_deps = set()

        for file in feature_files:
            rel_file = os.path.relpath(file, self.repo_path)
            if rel_file in self.imports_map:
                for imp in self.imports_map[rel_file]:
                    base_import = imp.split(".")[0]

                    # Check if import is a fundamental ML library
                    if self._is_fundamental_ml_library(imp):
                        fundamental_deps.add(imp)
                        continue

                    # Check if it's a standard library
                    if self._is_stdlib(base_import):
                        continue  # Don't count stdlib as external dependency

                    # Check if import starts with any of the base package names
                    is_internal = any(
                        imp.startswith(pkg) or base_import == pkg
                        for pkg in self.base_packages
                    )

                    if is_internal:
                        internal_deps.add(imp)
                    else:
                        external_deps.add(imp)

        return internal_deps, external_deps, fundamental_deps

    def _is_stdlib(self, package_name: str) -> bool:
        """Check if a package is part of Python standard library"""
        stdlib_modules = {
            "abc",
            "aifc",
            "argparse",
            "array",
            "ast",
            "asynchat",
            "asyncio",
            "asyncore",
            "atexit",
            "audioop",
            "base64",
            "bdb",
            "binascii",
            "binhex",
            "bisect",
            "builtins",
            "bz2",
            "calendar",
            "cgi",
            "cgitb",
            "chunk",
            "cmath",
            "cmd",
            "code",
            "codecs",
            "codeop",
            "collections",
            "colorsys",
            "compileall",
            "concurrent",
            "configparser",
            "contextlib",
            "contextvars",
            "copy",
            "copyreg",
            "crypt",
            "csv",
            "ctypes",
            "curses",
            "dataclasses",
            "datetime",
            "dbm",
            "decimal",
            "difflib",
            "dis",
            "distutils",
            "doctest",
            "email",
            "encodings",
            "enum",
            "errno",
            "faulthandler",
            "fcntl",
            "filecmp",
            "fileinput",
            "fnmatch",
            "formatter",
            "fractions",
            "ftplib",
            "functools",
            "gc",
            "getopt",
            "getpass",
            "gettext",
            "glob",
            "grp",
            "gzip",
            "hashlib",
            "heapq",
            "hmac",
            "html",
            "http",
            "imaplib",
            "imghdr",
            "imp",
            "importlib",
            "inspect",
            "io",
            "ipaddress",
            "itertools",
            "json",
            "keyword",
            "lib2to3",
            "linecache",
            "locale",
            "logging",
            "lzma",
            "mailbox",
            "mailcap",
            "marshal",
            "math",
            "mimetypes",
            "mmap",
            "modulefinder",
            "msilib",
            "msvcrt",
            "multiprocessing",
            "netrc",
            "nis",
            "nntplib",
            "numbers",
            "operator",
            "optparse",
            "os",
            "ossaudiodev",
            "parser",
            "pathlib",
            "pdb",
            "pickle",
            "pickletools",
            "pipes",
            "pkgutil",
            "platform",
            "plistlib",
            "poplib",
            "posix",
            "posixpath",
            "pprint",
            "profile",
            "pstats",
            "pty",
            "pwd",
            "py_compile",
            "pyclbr",
            "pydoc",
            "queue",
            "quopri",
            "random",
            "re",
            "readline",
            "reprlib",
            "resource",
            "rlcompleter",
            "runpy",
            "sched",
            "secrets",
            "select",
            "selectors",
            "shelve",
            "shlex",
            "shutil",
            "signal",
            "site",
            "smtpd",
            "smtplib",
            "sndhdr",
            "socket",
            "socketserver",
            "spwd",
            "sqlite3",
            "ssl",
            "stat",
            "statistics",
            "string",
            "stringprep",
            "struct",
            "subprocess",
            "sunau",
            "symbol",
            "symtable",
            "sys",
            "sysconfig",
            "syslog",
            "tabnanny",
            "tarfile",
            "telnetlib",
            "tempfile",
            "termios",
            "test",
            "textwrap",
            "threading",
            "time",
            "timeit",
            "tkinter",
            "token",
            "tokenize",
            "trace",
            "traceback",
            "tracemalloc",
            "tty",
            "turtle",
            "turtledemo",
            "types",
            "typing",
            "unicodedata",
            "unittest",
            "urllib",
            "uu",
            "uuid",
            "venv",
            "warnings",
            "wave",
            "weakref",
            "webbrowser",
            "winreg",
            "winsound",
            "wsgiref",
            "xdrlib",
            "xml",
            "xmlrpc",
            "zipapp",
            "zipfile",
            "zipimport",
            "zlib",
        }

        return package_name in stdlib_modules

    def _is_fundamental_ml_library(self, package_name: str) -> bool:
        """Check if package is a fundamental ML/scientific library"""
        fundamental_libs = {
            # Deep Learning Frameworks
            "torch",
            "pytorch",
            "tensorflow",
            "tf",
            "keras",
            "jax",
            "flax",
            # Scientific Computing
            "numpy",
            "np",
            "scipy",
            "pandas",
            "pd",
            # ML Libraries
            "sklearn",
            "scikit-learn",
            "xgboost",
            "lightgbm",
            "catboost",
            # Visualization
            "matplotlib",
            "seaborn",
            "plotly",
            "bokeh",
            # Data Processing
            "cv2",
            "opencv",
            "PIL",
            "pillow",
            "skimage",
            "imageio",
            # NLP
            "transformers",
            "huggingface",
            "nltk",
            "spacy",
            "gensim",
            # Others
            "gym",
            "gymnasium",
            "accelerate",
            "datasets",
            "tokenizers",
            "wandb",
            "tensorboard",
            "mlflow",
        }

        base_package = package_name.split(".")[0].lower()
        return base_package in fundamental_libs

    def calculate_coupling(self, feature_files: List[str]) -> float:
        """
        Calculates coupling based on the concentration of dependencies in the graph.
        A lower score (more distributed dependencies) is better.
        """
        if self.module_graph.number_of_nodes() == 0:
            return 0.0

        # Use in-degree centrality: measures how many other files depend on a given file
        centrality = nx.in_degree_centrality(self.module_graph)

        if not centrality or not centrality.values():
            return 0.0

        # The coupling score is based on the HIGHEST centrality value
        max_centrality = max(centrality.values())
        return max_centrality

    def calculate_cohesion(self, feature_files: List[str]) -> float:
        """Calculate cohesion based on internal function call density"""
        if not feature_files:
            return 0.0

        # Count internal function calls
        internal_calls = 0
        total_functions = 0
        all_repo_functions = set()

        # Collect all function definitions
        for analysis in self.file_analyses.values():
            for func_name in analysis.get("functions", []):
                all_repo_functions.add(func_name)

        total_functions = len(all_repo_functions)
        if total_functions == 0:
            return 0.0

        # Count internal function calls
        for analysis in self.file_analyses.values():
            for called_func in analysis.get("function_calls", set()):
                base_func = called_func.split(".")[-1]
                if base_func in all_repo_functions:
                    internal_calls += 1

        # Better scaling: use logarithmic ratio
        if internal_calls == 0:
            return 0.0

        ratio = internal_calls / total_functions
        # Use log scale to prevent saturation
        cohesion_score = math.log(1 + ratio) / math.log(1 + 10)  # Normalized to 0-1
        return min(cohesion_score, 1.0)

    def get_dependency_depth(self, feature_files: List[str]) -> int:
        """Calculate maximum dependency depth"""
        if not self.module_graph.nodes():
            return 0

        max_depth = 0

        for file in feature_files:
            rel_file = os.path.relpath(file, self.repo_path)
            if rel_file in self.module_graph:
                try:
                    paths = nx.single_source_shortest_path_length(
                        self.module_graph, rel_file
                    )
                    max_depth = max(max_depth, max(paths.values()) if paths else 0)
                except:
                    pass

        return max_depth

    def calculate_modularity(self) -> float:
        """Calculates the modularity of the dependency graph."""
        if not HAS_LOUVAIN:
            return 0.0

        if (
            self.module_graph.number_of_nodes() < 2
            or self.module_graph.number_of_edges() == 0
        ):
            return 0.0

        # Convert to undirected for Louvain
        undirected_graph = self.module_graph.to_undirected()

        try:
            partition = community_louvain.best_partition(undirected_graph)
            modularity_score = community_louvain.modularity(partition, undirected_graph)
            return modularity_score
        except Exception:
            return 0.0

    def calculate_max_inheritance_depth(self) -> int:
        """Calculate the maximum inheritance depth in the codebase"""
        max_depth = 0

        for analysis in self.file_analyses.values():
            for cls in analysis.get("classes", []):
                depth = len(cls.get("bases", []))
                max_depth = max(max_depth, depth)

        return max_depth

    def detect_registry_pattern_usage(self) -> int:
        """Detect usage of registry/factory patterns"""
        registry_indicators = [
            "register",
            "registry",
            "register_module",
            "REGISTRY",
            "build_from_cfg",
            "build_",
            "Registry",
        ]

        count = 0

        for analysis in self.file_analyses.values():
            # Check decorators
            for decorator in analysis.get("decorators", []):
                if any(
                    indicator.lower() in decorator.lower()
                    for indicator in registry_indicators
                ):
                    count += 1

            # Check class names
            for cls in analysis.get("classes", []):
                if any(
                    indicator.lower() in cls["name"].lower()
                    for indicator in registry_indicators
                ):
                    count += 1

            # Check function names
            for func in analysis.get("functions", []):
                if any(
                    indicator.lower() in func.lower()
                    for indicator in registry_indicators
                ):
                    count += 1

        return count

    def detect_config_system(self) -> bool:
        """Detect if the repo uses a complex configuration system"""
        config_indicators = [
            "Config",
            "ConfigDict",
            "config",
            "cfg",
            "register_module",
            "Registry",
            "CONFIGS",
            "mmcv.Config",
            "DictConfig",
            "OmegaConf",
        ]

        # Check imports
        for imports in self.imports_map.values():
            for imp in imports:
                if any(
                    indicator.lower() in imp.lower() for indicator in config_indicators
                ):
                    return True

        # Check class names
        for analysis in self.file_analyses.values():
            for cls in analysis.get("classes", []):
                if any(
                    indicator.lower() in cls["name"].lower()
                    for indicator in config_indicators
                ):
                    return True

        return False


class GitHubRepoBatchAnalyzer:
    """Batch analyzer for multiple GitHub repositories"""

    def __init__(self, input_csv: str, output_csv: str = "extractability_report.csv"):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.results = []

    def extract_repo_name(self, url: str) -> str:
        """Extract repository name from GitHub URL"""
        patterns = [
            r"github\.com/([^/]+/[^/]+?)(?:\.git)?$",
            r"github\.com/([^/]+/[^/]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return url

    def clone_repo(self, repo_url: str, target_dir: str) -> bool:
        """Clone a GitHub repository"""
        try:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)

            result = subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, target_dir],
                capture_output=True,
                text=True,
                timeout=300,
            )

            return result.returncode == 0
        except Exception as e:
            print(f"Error cloning {repo_url}: {e}")
            return False

    def analyze_repo(self, repo_url: str) -> ExtractabilityMetrics:
        """Analyze a single repository"""
        repo_name = self.extract_repo_name(repo_url)
        print(f"\nAnalyzing: {repo_name}")

        temp_dir = tempfile.mkdtemp(prefix="repo_analysis_")

        try:
            print(f"  Cloning repository...")
            if not self.clone_repo(repo_url, temp_dir):
                return self._create_error_metrics(
                    repo_name, repo_url, "Failed to clone repository"
                )

            analyzer = DependencyAnalyzer(temp_dir)

            print(f"  Finding Python files...")
            python_files = analyzer.find_all_python_files()

            if not python_files:
                return self._create_error_metrics(
                    repo_name,
                    repo_url,
                    "No Python files found in repository",
                    status="no_python_files",
                    total_files=len(list(Path(temp_dir).rglob("*"))),
                )

            print(f"  Found {len(python_files)} Python files")
            print(f"  Building dependency graph...")

            analyzer.build_dependency_graph(python_files)

            print(f"  Calculating metrics...")
            internal, external, fundamental = analyzer.get_internal_vs_external_deps(
                python_files
            )

            print(f"    - Internal dependencies: {len(internal)}")
            print(f"    - External dependencies: {len(external)}")
            print(f"    - Base packages detected: {analyzer.base_packages}")

            coupling = analyzer.calculate_coupling(python_files)
            cohesion = analyzer.calculate_cohesion(python_files)
            depth = analyzer.get_dependency_depth(python_files)
            modularity = analyzer.calculate_modularity()
            max_inheritance = analyzer.calculate_max_inheritance_depth()
            registry_usage = analyzer.detect_registry_pattern_usage()
            has_config_system = analyzer.detect_config_system()

            # Calculate complexity metrics
            complexities = list(analyzer.file_complexities.values())
            avg_complexity = (
                sum(complexities) / len(complexities) if complexities else 0
            )
            max_complexity = max(complexities) if complexities else 0

            # Count cross-module calls
            cross_module_calls = sum(
                len(analyzer.function_calls.get(os.path.relpath(f, temp_dir), set()))
                for f in python_files
            )

            # Count classes and functions
            total_classes = sum(
                len(analysis.get("classes", []))
                for analysis in analyzer.file_analyses.values()
            )
            total_functions = sum(
                len(analysis.get("functions", []))
                for analysis in analyzer.file_analyses.values()
            )
            # Normalized scores
            complexity_score = 1 - (1 / (1 + math.log(1 + avg_complexity)))
            inheritance_score = min(max_inheritance / 5.0, 1.0)
            
            # SIZE-AWARE ADJUSTMENTS
            num_files = len(python_files)
            
            # 1. Small repo penalty (< 100 files get penalized)
            if num_files < 100:
                size_penalty = (100 - num_files) / 400  # Up to -0.25 penalty
            else:
                size_penalty = 0.0
            
            # 2. Registry score adjusted by repo size
            # Small repos: registries are worse (less justified)
            # Large repos: some registry use is acceptable for organization
            if num_files < 200:
                registry_score = min(registry_usage / 50.0, 1.0)  # Harsh for small repos
            elif num_files < 500:
                registry_score = min(registry_usage / 100.0, 1.0)  # Moderate
            else:
                registry_score = min(registry_usage / 150.0, 1.0)  # Forgiving for large repos
            
            # 3. Config penalty based on registry interaction
            # Config + heavy registries = bad (MMSeg style)
            # Config + light registries = okay (organized style)
            if has_config_system:
                if registry_usage > 200:
                    config_penalty = 0.20  # Heavy penalty for config + registries
                elif registry_usage > 50:
                    config_penalty = 0.10  # Moderate penalty
                else:
                    config_penalty = 0.05  # Light penalty - configs alone aren't terrible
            else:
                config_penalty = 0.0
            
            # 4. Internal dependency bonus (size-adjusted)
            internal_ratio = len(internal) / (len(internal) + len(external) + 1)
            # Large repos should have MORE internal deps to be self-contained
            if num_files > 500:
                expected_internal_ratio = 0.6  # Expect at least 60% internal for large repos
                internal_bonus = max(0, (internal_ratio - expected_internal_ratio)) * 0.20
            elif num_files > 200:
                expected_internal_ratio = 0.5  # Expect at least 50% internal
                internal_bonus = max(0, (internal_ratio - expected_internal_ratio)) * 0.15
            else:
                internal_bonus = internal_ratio * 0.10  # Small repos: linear bonus
            
            # BALANCED WEIGHTS (redistributed from 40% modularity)
            extractability = 100 * (
                (1 - coupling) * 0.25 +           # Increased from 0.20
                modularity * 0.35 +               # Reduced from 0.40 - still most important
                cohesion * 0.10 +                 # Increased from 0.05
                (1 - complexity_score) * 0.15 +   # Increased from 0.10
                (1 - inheritance_score) * 0.05 +  # Same
                (1 - registry_score) * 0.10 -     # Reduced from 0.20 (now context-aware)
                config_penalty -                  # Variable instead of flat 0.20
                size_penalty +                    # NEW penalty for tiny repos
                internal_bonus                    # Size-adjusted bonus
            )
            
            # Ensure score is between 0 and 100
            extractability = max(0, min(100, extractability))

            total_files = len(list(Path(temp_dir).rglob("*")))

            print(f"  ✓ Analysis complete - Score: {extractability:.2f}/100")

            return ExtractabilityMetrics(
                repo_name=repo_name,
                repo_url=repo_url,
                total_files=total_files,
                total_python_files=len(python_files),
                internal_dependencies=len(internal),
                external_dependencies=len(external),
                fundamental_ml_dependencies=len(fundamental),
                cross_module_calls=cross_module_calls,
                coupling_score=round(coupling, 3),
                cohesion_score=round(cohesion, 3),
                modularity_score=round(modularity, 3),
                complexity_score=round(complexity_score, 3),
                dependency_depth=depth,
                max_inheritance_depth=max_inheritance,
                registry_pattern_usage=registry_usage,
                has_config_system=has_config_system,
                extractability_score=round(extractability, 2),
                avg_file_complexity=round(avg_complexity, 2),
                max_file_complexity=int(max_complexity),
                total_classes=total_classes,
                total_functions=total_functions,
                analysis_status="success",
            )

        except Exception as e:
            print(f"  ✗ Error analyzing repository: {e}")
            import traceback

            traceback.print_exc()
            return self._create_error_metrics(repo_name, repo_url, str(e))

        finally:
            try:
                if os.path.exists(temp_dir):
                    # Windows fix: mark files as writable before deletion
                    for root, dirs, files in os.walk(temp_dir):
                        for dir in dirs:
                            os.chmod(os.path.join(root, dir), 0o777)
                        for file in files:
                            os.chmod(os.path.join(root, file), 0o777)
                    shutil.rmtree(temp_dir)
                    print(f"  Cleaned up temporary files")
            except Exception as e:
                print(f"  Warning: Could not delete {temp_dir}: {e}")

    def _create_error_metrics(
        self,
        repo_name: str,
        repo_url: str,
        error_msg: str,
        status: str = "error",
        total_files: int = 0,
    ) -> ExtractabilityMetrics:
        """Helper to create error metrics"""
        return ExtractabilityMetrics(
            repo_name=repo_name,
            repo_url=repo_url,
            total_files=total_files,
            total_python_files=0,
            internal_dependencies=0,
            external_dependencies=0,
            fundamental_ml_dependencies=0,
            cross_module_calls=0,
            coupling_score=0.0,
            cohesion_score=0.0,
            modularity_score=0.0,
            complexity_score=0.0,
            dependency_depth=0,
            max_inheritance_depth=0,
            registry_pattern_usage=0,
            has_config_system=False,
            extractability_score=0.0,
            avg_file_complexity=0.0,
            max_file_complexity=0.0,
            total_classes=0,
            total_functions=0,
            analysis_status=status,
            error_message=error_msg,
        )

    def process_batch(self):
        """Process all repositories from CSV"""
        print(f"Reading repositories from {self.input_csv}...")

        repos = []
        try:
            with open(self.input_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    url = (
                        row.get("url")
                        or row.get("repo_url")
                        or row.get("github_url")
                        or row.get("URL")
                    )
                    if url:
                        repos.append(url.strip())
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return

        print(f"Found {len(repos)} repositories to analyze\n")
        print("=" * 70)

        for i, repo_url in enumerate(repos, 1):
            print(f"\n[{i}/{len(repos)}] Processing: {repo_url}")
            print("-" * 70)

            metrics = self.analyze_repo(repo_url)
            self.results.append(metrics)

        self.generate_csv_report()
        print(f"\n{'=' * 70}")
        print(f"Analysis complete! Report saved to: {self.output_csv}")
        print(
            f"Successfully analyzed: {sum(1 for r in self.results if r.analysis_status == 'success')}/{len(repos)}"
        )

    def generate_csv_report(self):
        """Generate CSV report with all metrics"""
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            if not self.results:
                return

            writer = csv.DictWriter(
                f, fieldnames=ExtractabilityMetrics.get_csv_headers()
            )
            writer.writeheader()

            for result in self.results:
                writer.writerow(result.to_dict())


# Example usage
if __name__ == "__main__":
    """
    Input CSV format (any of these column names work):
    - 'url' or 'repo_url' or 'github_url' or 'URL'

    Example CSV content:
    url
    https://github.com/user/repo1
    https://github.com/user/repo2
    """

    # Configure input and output files
    INPUT_CSV = "repos_to_analyze.csv"
    OUTPUT_CSV = "extractability_analysis_results_v7.csv"

    # Run batch analysis
    analyzer = GitHubRepoBatchAnalyzer(INPUT_CSV, OUTPUT_CSV)
    analyzer.process_batch()
