"""
Batch Feature Extractability Analyzer for GitHub Repositories
Reads repos from CSV, analyzes them, and generates a report
"""

import ast
import os
import json
import csv
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, asdict, fields
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
from pathlib import Path
import re
import networkx as nx
import math
from community import community_louvain


@dataclass
class ExtractabilityMetrics:
    """Metrics for measuring feature extractability"""
    repo_name: str
    repo_url: str
    total_files: int
    total_python_files: int
    internal_dependencies: int
    external_dependencies: int
    fundamental_ml_dependencies: int  # New field
    cross_module_calls: int
    coupling_score: float  # 0-1, lower is better
    cohesion_score: float  # 0-1, higher is better
    complexity_score: float
    dependency_depth: int
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
        
    def analyze_file(self, filepath: str) -> Dict:
        """Analyze a single Python file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if not content.strip():
                    return {}
                tree = ast.parse(content, filename=filepath)
        except (SyntaxError, UnicodeDecodeError, Exception) as e:
            return {}
        
        relative_path = os.path.relpath(filepath, self.repo_path)
        
        analysis = {
            'imports': set(),
            'function_calls': set(),
            'classes': [],
            'functions': [],
            'complexity': 0
        }
        
        for node in ast.walk(tree):
            # Track imports
            if isinstance(node, ast.Import):
                for name in node.names:
                    analysis['imports'].add(name.name)
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    analysis['imports'].add(node.module)
            
            # Track class definitions
            elif isinstance(node, ast.ClassDef):
                analysis['classes'].append({
                    'name': node.name,
                    'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                    'bases': [self._get_name(base) for base in node.bases]
                })
            
            # Track function definitions
            elif isinstance(node, ast.FunctionDef):
                analysis['functions'].append(node.name)
                # Measure complexity (simplified cyclomatic)
                analysis['complexity'] += self._calculate_complexity(node)
            
            # Track function calls
            elif isinstance(node, ast.Call):
                call_name = self._get_name(node.func)
                if call_name:
                    analysis['function_calls'].add(call_name)
        
        self.imports_map[relative_path] = analysis['imports']
        self.function_calls[relative_path] = analysis['function_calls']
        self.file_complexities[relative_path] = analysis['complexity']
        self.file_analyses[relative_path] = analysis
        
        return analysis
    
    def _get_name(self, node) -> str:
        """Extract name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
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
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', 'env', '.venv', 'node_modules', '.tox']]
            
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    python_files.append(full_path)
        
        return python_files
    
    def build_dependency_graph(self, feature_files: List[str]):
        """Build directed graph of dependencies"""
        for filepath in feature_files:
            if os.path.exists(filepath) and filepath.endswith('.py'):
                self.analyze_file(filepath)
        
        # Build graph edges
        for file, imports in self.imports_map.items():
            for imp in imports:
                self.module_graph.add_edge(file, imp)
    
    def get_internal_vs_external_deps(self, feature_files: List[str]) -> Tuple[Set, Set]:
        """Separate internal (within repo) from external dependencies"""
        feature_set = set(os.path.relpath(f, self.repo_path) for f in feature_files)
        internal_deps = set()
        external_deps = set()
        fundamental_deps = set()  # Track fundamental ML libraries separately
        
        for file in feature_files:
            rel_file = os.path.relpath(file, self.repo_path)
            if rel_file in self.imports_map:
                for imp in self.imports_map[rel_file]:
                    # Check if import is a fundamental ML library
                    if self._is_fundamental_ml_library(imp):
                        fundamental_deps.add(imp)
                        continue
                    
                    # Check if import is a standard library or external package
                    is_external = self._is_external_package(imp)
                    
                    # Check if it's another file in the repo
                    is_internal = any(
                        imp.replace('.', '/') in f or 
                        f.startswith(imp.replace('.', '/'))
                        for f in feature_set
                    )
                    
                    if is_external:
                        external_deps.add(imp)
                    else:
                        internal_deps.add(imp)
        
        # Return internal, external (excluding fundamental libs), and fundamental separately
        return internal_deps, external_deps, fundamental_deps
    
    def _is_external_package(self, package_name: str) -> bool:
        """Check if a package is external (not in the repo)"""
        # Common external packages
        stdlib_modules = {
            'os', 'sys', 'json', 'csv', 're', 'math', 'random', 'datetime',
            'collections', 'itertools', 'functools', 'pathlib', 'typing',
            'unittest', 'logging', 'argparse', 'subprocess', 'threading',
            'multiprocessing', 'socket', 'http', 'urllib', 'email', 'io'
        }
        
        base_package = package_name.split('.')[0]
        return base_package in stdlib_modules or base_package not in self.imports_map
    
    def _is_fundamental_ml_library(self, package_name: str) -> bool:
        """Check if package is a fundamental ML/scientific library"""
        fundamental_libs = {
            # Deep Learning Frameworks
            'torch', 'pytorch', 'tensorflow', 'tf', 'keras', 'jax', 'flax',
            # Scientific Computing
            'numpy', 'np', 'scipy', 'pandas', 'pd',
            # ML Libraries
            'sklearn', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost',
            # Visualization
            'matplotlib', 'seaborn', 'plotly', 'bokeh',
            # Data Processing
            'cv2', 'opencv', 'PIL', 'pillow', 'skimage', 'imageio',
            # NLP
            'transformers', 'huggingface', 'nltk', 'spacy', 'gensim',
            # Others
            'gym', 'gymnasium', 'accelerate', 'datasets', 'tokenizers'
        }
        
        base_package = package_name.split('.')[0].lower()
        return base_package in fundamental_libs
    
    def calculate_coupling(self, feature_files: List[str]) -> float:
        """
        Calculates coupling based on the concentration of dependencies in the graph.
        A lower score (more distributed dependencies) is better.
        """
        if self.module_graph.number_of_nodes() == 0:
            return 0.0

        # Use in-degree centrality: it measures how many other files depend on a given file.
        # This is a direct measure of a module's responsibility and potential impact.
        centrality = nx.in_degree_centrality(self.module_graph)
        
        if not centrality:
            return 0.0

        # The coupling score is based on the HIGHEST centrality value found in the graph.
        # A high max centrality indicates a "super-connector" or god object, which is a sign of tight coupling.
        max_centrality = max(centrality.values())
        
        # We can use the max value directly as the score. It's already normalized between 0 and 1.
        # A project with at least one file that many others depend on will get a higher coupling score.
        return max_centrality
    
    def calculate_cohesion(self, feature_files: List[str]) -> float:
        if not feature_files:
            return 0.0

        # Count internal function calls (a better proxy for cohesion than imports)
        internal_calls = 0
        total_functions = 0
        all_repo_functions = set()

        # First, collect all function definitions across all files
        for analysis in self.file_analyses.values():
            for func_name in analysis.get('functions', []):
                all_repo_functions.add(func_name)
        
        total_functions = len(all_repo_functions)
        if total_functions == 0:
            return 0.0

        # Now, count how many function calls are to other functions within the repo
        for analysis in self.file_analyses.values():
            for called_func in analysis.get('function_calls', set()):
                # Check if the base of the call (e.g., 'my_func' in 'utils.my_func') is a known function
                if called_func.split('.')[-1] in all_repo_functions:
                    internal_calls += 1
                    
        # Normalize by the total number of functions. This is more stable.
        # We add 1 to avoid division by zero and to scale the result
        cohesion_score = (internal_calls / (total_functions + 1)) / 10 # scale it down to keep it in a 0-1 range
        return min(cohesion_score, 1.0) # Cap at 1.0
    
    def get_dependency_depth(self, feature_files: List[str]) -> int:
        """Calculate maximum dependency depth"""
        if not self.module_graph.nodes():
            return 0
            
        max_depth = 0
        
        for file in feature_files:
            rel_file = os.path.relpath(file, self.repo_path)
            if rel_file in self.module_graph:
                try:
                    # Find longest path from this node
                    paths = nx.single_source_shortest_path_length(self.module_graph, rel_file)
                    max_depth = max(max_depth, max(paths.values()) if paths else 0)
                except:
                    pass
        
        return max_depth
    def calculate_modularity(self) -> float:
        """Calculates the modularity of the dependency graph."""
        if self.module_graph.number_of_nodes() < 2 or self.module_graph.number_of_edges() == 0:
            return 0.0
        
        # The graph must be undirected for the Louvain method
        undirected_graph = self.module_graph.to_undirected()
        
        try:
            # Find the best partition (communities)
            partition = community_louvain.best_partition(undirected_graph)
            # Calculate the modularity of that partition
            modularity_score = community_louvain.modularity(partition, undirected_graph)
            return modularity_score
        except Exception:
            return 0.0


class GitHubRepoBatchAnalyzer:
    """Batch analyzer for multiple GitHub repositories"""
    
    def __init__(self, input_csv: str, output_csv: str = "extractability_report.csv"):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.results = []
    
    def extract_repo_name(self, url: str) -> str:
        """Extract repository name from GitHub URL"""
        # Handle various GitHub URL formats
        patterns = [
            r'github\.com/([^/]+/[^/]+?)(?:\.git)?$',
            r'github\.com/([^/]+/[^/]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return url
    
    def clone_repo(self, repo_url: str, target_dir: str) -> bool:
        """Clone a GitHub repository"""
        try:
            # Ensure target directory doesn't exist
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            
            # Clone with shallow depth to save time and space
            result = subprocess.run(
                ['git', 'clone', '--depth', '1', repo_url, target_dir],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return result.returncode == 0
        except Exception as e:
            print(f"Error cloning {repo_url}: {e}")
            return False
    
    def analyze_repo(self, repo_url: str) -> ExtractabilityMetrics:
        """Analyze a single repository"""
        repo_name = self.extract_repo_name(repo_url)
        print(f"\nAnalyzing: {repo_name}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="repo_analysis_")
        
        try:
            # Clone repository
            print(f"  Cloning repository...")
            if not self.clone_repo(repo_url, temp_dir):
                return ExtractabilityMetrics(
                    repo_name=repo_name,
                    repo_url=repo_url,
                    total_files=0,
                    total_python_files=0,
                    internal_dependencies=0,
                    external_dependencies=0,
                    fundamental_ml_dependencies=0,
                    cross_module_calls=0,
                    coupling_score=0.0,
                    cohesion_score=0.0,
                    complexity_score=0.0,
                    dependency_depth=0,
                    extractability_score=0.0,
                    avg_file_complexity=0.0,
                    max_file_complexity=0.0,
                    total_classes=0,
                    total_functions=0,
                    analysis_status='error',
                    error_message='Failed to clone repository'
                )
            
            # Initialize analyzer
            analyzer = DependencyAnalyzer(temp_dir)
            
            # Find all Python files
            print(f"  Finding Python files...")
            python_files = analyzer.find_all_python_files()
            
            if not python_files:
                return ExtractabilityMetrics(
                    repo_name=repo_name,
                    repo_url=repo_url,
                    total_files=len(list(Path(temp_dir).rglob('*'))),
                    total_python_files=0,
                    internal_dependencies=0,
                    external_dependencies=0,
                    fundamental_ml_dependencies=0,
                    cross_module_calls=0,
                    coupling_score=0.0,
                    cohesion_score=0.0,
                    complexity_score=0.0,
                    dependency_depth=0,
                    extractability_score=0.0,
                    avg_file_complexity=0.0,
                    max_file_complexity=0.0,
                    total_classes=0,
                    total_functions=0,
                    analysis_status='no_python_files',
                    error_message='No Python files found in repository'
                )
            
            print(f"  Found {len(python_files)} Python files")
            print(f"  Building dependency graph...")
            
            # Build dependency graph
            analyzer.build_dependency_graph(python_files)
            
            # Calculate metrics
            print(f"  Calculating metrics...")
            internal, external, fundamental = analyzer.get_internal_vs_external_deps(python_files)
            coupling = analyzer.calculate_coupling(python_files)
            cohesion = analyzer.calculate_cohesion(python_files)
            depth = analyzer.get_dependency_depth(python_files)
            
            # Calculate complexity metrics
            complexities = list(analyzer.file_complexities.values())
            avg_complexity = sum(complexities) / len(complexities) if complexities else 0
            max_complexity = max(complexities) if complexities else 0
            
            # Count cross-module calls
            cross_module_calls = sum(
                len(analyzer.function_calls.get(os.path.relpath(f, temp_dir), set()))
                for f in python_files
            )
            
            # Count classes and functions
            total_classes = sum(
                len(analysis.get('classes', []))
                for analysis in analyzer.file_analyses.values()
            )
            total_functions = sum(
                len(analysis.get('functions', []))
                for analysis in analyzer.file_analyses.values()
            )
            
            # Normalized complexity score
            complexity_score = 1 - (1 / (1 + math.log(1 + avg_complexity)))

            modularity = analyzer.calculate_modularity()
            
            # Calculate overall extractability score (0-100)
            extractability = 100 * (
                (1 - coupling) * 0.4 +
                modularity * 0.35 + # Give modularity a high weight
                (1 - complexity_score) * 0.2 +
                (1 / (depth + 1)) * 0.05 # Reduce depth's weight
            )
            
            total_files = len(list(Path(temp_dir).rglob('*')))
            
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
                complexity_score=round(complexity_score, 3),
                dependency_depth=depth,
                extractability_score=round(extractability, 2),
                avg_file_complexity=round(avg_complexity, 2),
                max_file_complexity=int(max_complexity),
                total_classes=total_classes,
                total_functions=total_functions,
                analysis_status='success'
            )
            
        except Exception as e:
            print(f"  ✗ Error analyzing repository: {e}")
            return ExtractabilityMetrics(
                repo_name=repo_name,
                repo_url=repo_url,
                total_files=0,
                total_python_files=0,
                internal_dependencies=0,
                external_dependencies=0,
                fundamental_ml_dependencies=0,
                cross_module_calls=0,
                coupling_score=0.0,
                cohesion_score=0.0,
                complexity_score=0.0,
                dependency_depth=0,
                extractability_score=0.0,
                avg_file_complexity=0.0,
                max_file_complexity=0.0,
                total_classes=0,
                total_functions=0,
                analysis_status='error',
                error_message=str(e)
            )
        
        finally:
            # Clean up: delete the repository
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"  Cleaned up temporary files")
            except Exception as e:
                print(f"  Warning: Could not delete {temp_dir}: {e}")
    
    def process_batch(self):
        """Process all repositories from CSV"""
        print(f"Reading repositories from {self.input_csv}...")
        
        repos = []
        try:
            with open(self.input_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Support different column names
                    url = row.get('url') or row.get('repo_url') or row.get('github_url') or row.get('URL')
                    if url:
                        repos.append(url.strip())
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return
        
        print(f"Found {len(repos)} repositories to analyze\n")
        print("=" * 70)
        
        # Analyze each repository
        for i, repo_url in enumerate(repos, 1):
            print(f"\n[{i}/{len(repos)}] Processing: {repo_url}")
            print("-" * 70)
            
            metrics = self.analyze_repo(repo_url)
            self.results.append(metrics)
        
        # Generate report
        self.generate_csv_report()
        print(f"\n{'=' * 70}")
        print(f"Analysis complete! Report saved to: {self.output_csv}")
        print(f"Successfully analyzed: {sum(1 for r in self.results if r.analysis_status == 'success')}/{len(repos)}")
    
    def generate_csv_report(self):
        """Generate CSV report with all metrics"""
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            if not self.results:
                return
            
            writer = csv.DictWriter(f, fieldnames=ExtractabilityMetrics.get_csv_headers())
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
    OUTPUT_CSV = "extractability_analysis_results_v4.csv"
    
    # Run batch analysis
    analyzer = GitHubRepoBatchAnalyzer(INPUT_CSV, OUTPUT_CSV)
    analyzer.process_batch()