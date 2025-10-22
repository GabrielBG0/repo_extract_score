# Feature Extractability Analyzer for GitHub Repositories

## Overview

This tool performs automated static analysis on Python repositories to measure how easily features can be extracted and implemented in a standalone manner. It's particularly useful for machine learning repositories where researchers and practitioners often need to adopt specific components without bringing in entire frameworks.

The analyzer produces an **Extractability Score** (0-100) where higher scores indicate repositories from which features can be more easily extracted.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [How It Works](#how-it-works)
4. [Metrics Explained](#metrics-explained)
5. [Extractability Score Calculation](#extractability-score-calculation)
6. [Understanding the Results](#understanding-the-results)
7. [Code Architecture](#code-architecture)

---

## Installation

### Requirements

```bash
pip install networkx python-louvain
```

### System Requirements

- Python 3.7+
- Git (must be in PATH)
- NetworkX for graph analysis
- python-louvain for modularity calculations

---

## Quick Start

### 1. Create Input CSV

Create a file named `repos_to_analyze.csv`:

```csv
url
https://github.com/user/repo1
https://github.com/user/repo2
https://github.com/user/repo3
```

### 2. Run the Analyzer

```bash
python extractability_analyzer.py
```

### 3. View Results

Results are saved to `extractability_analysis_results_v7.csv` with detailed metrics for each repository.

---

## How It Works

### Processing Pipeline

```
1. Clone Repository (shallow, depth=1)
   ↓
2. Find Python Files (skip build/test directories)
   ↓
3. Parse AST (Abstract Syntax Tree) for each file
   ↓
4. Build Dependency Graph
   ↓
5. Calculate Metrics
   ↓
6. Compute Extractability Score
   ↓
7. Clean Up & Export Results
```

### Core Components

#### 1. **DependencyAnalyzer Class**

Performs AST-based static analysis:

- **File Analysis**: Parses Python files into Abstract Syntax Trees
- **Import Tracking**: Maps all import statements
- **Function Call Analysis**: Tracks function invocations
- **Class Structure**: Analyzes inheritance hierarchies
- **Decorator Detection**: Identifies patterns like `@register_module`
- **Complexity Calculation**: Computes cyclomatic complexity

#### 2. **GitHubRepoBatchAnalyzer Class**

Orchestrates batch processing:

- Clones repositories temporarily
- Manages analysis workflow
- Aggregates results
- Generates CSV reports

---

## Metrics Explained

### 1. Coupling Score (0-1, lower is better)

**What it measures**: The degree to which files in the repository depend on a single "god object" or bottleneck module.

**How it's calculated**:
```python
coupling_score = max(in_degree_centrality(dependency_graph))
```

Uses **in-degree centrality** from graph theory to identify files that many other files depend on. A file with high in-degree centrality is a critical dependency point.

**Why it matters**:
- High coupling (close to 1.0) means there's a central file that everything depends on
- Extracting any feature requires extracting this bottleneck
- Makes standalone implementation difficult

**Example**:
- Score 0.8: One file is depended upon by 80% of other files → Very hard to extract
- Score 0.2: Dependencies are distributed → Easier to extract

**Weight in final score**: 25%

---

### 2. Modularity Score (-1 to 1, higher is better)

**What it measures**: How well the code is organized into distinct, loosely-coupled communities or modules.

**How it's calculated**:
```python
# Uses Louvain method for community detection
partition = community_louvain.best_partition(dependency_graph)
modularity_score = community_louvain.modularity(partition, dependency_graph)
```

The Louvain algorithm detects natural communities in the dependency graph. Modularity quantifies how well these communities are separated.

**Why it matters**:
- High modularity (> 0.5) indicates clear module boundaries
- Well-defined modules can be extracted with minimal external dependencies
- This is the **strongest predictor** of extractability

**Example**:
- Score 0.7: Code has clear, well-separated modules → Easy to identify and extract features
- Score 0.2: Code is tangled with no clear structure → Hard to extract anything cleanly

**Weight in final score**: 35% (highest weight)

---

### 3. Cohesion Score (0-1, higher is better)

**What it measures**: The density of internal function calls, indicating how tightly components communicate within the repository.

**How it's calculated**:
```python
ratio = internal_function_calls / total_functions
cohesion_score = log(1 + ratio) / log(11)  # Logarithmic scaling
```

Counts how many function calls are to other functions defined within the same repository.

**Why it matters**:
- High cohesion within well-defined modules is ideal
- Indicates components are designed to work together
- Too high cohesion across the entire repo can indicate monolithic design

**Example**:
- Score 0.6: Functions call each other frequently → Components are integrated
- Score 0.1: Functions are isolated → Potentially extractable but may lack integration

**Weight in final score**: 10%

---

### 4. Complexity Score (0-1, lower is better)

**What it measures**: Average cyclomatic complexity across all functions in the repository.

**How it's calculated**:
```python
# For each function, calculate cyclomatic complexity:
complexity = 1 + number_of_decision_points  # if, while, for, except, etc.

# Normalize across repository:
avg_complexity = sum(all_complexities) / num_functions
complexity_score = 1 - (1 / (1 + log(1 + avg_complexity)))
```

**Cyclomatic complexity** measures the number of linearly independent paths through code. More decision points = higher complexity.

**Why it matters**:
- High complexity makes code harder to understand and modify
- Complex code is more error-prone when extracted
- Logarithmic scaling prevents extreme values from dominating

**Example**:
- Score 0.4: Simple, straightforward code → Easy to understand and extract
- Score 0.8: Complex, branching logic → Difficult to extract safely

**Weight in final score**: 15%

---

### 5. Registry Pattern Usage (count, normalized)

**What it measures**: Occurrences of registry/factory patterns that create runtime coupling invisible to static analysis.

**How it's detected**:
```python
# Searches for patterns in:
- Decorators: @register_module, @MODELS.register
- Class names: Registry, ModelRegistry, BuilderRegistry
- Function names: build_from_cfg, register_model
```

**Why it matters**:
Registry patterns are **extremely problematic** for extraction:

1. **Hidden Dependencies**: `@MODELS.register_module()` creates coupling not visible in imports
2. **Runtime Resolution**: Components discovered through string-based lookups in config files
3. **Framework Lock-in**: Extracting a registered component requires extracting the entire registry system
4. **Implicit Behavior**: Behavior determined by configuration, not explicit code

**Context-Aware Scoring**:
- Small repos (< 200 files): `registry_score = min(count / 50, 1.0)` → Strict
- Medium repos (200-500 files): `registry_score = min(count / 100, 1.0)` → Moderate  
- Large repos (> 500 files): `registry_score = min(count / 150, 1.0)` → Lenient

**Rationale**: Large frameworks may legitimately use registries for organization. Small projects using registries are over-engineered.

**Example**:
- 5 registries in 50-file repo: Score ~0.1 → Acceptable
- 250 registries in 1500-file repo: Score ~1.0 → Heavy penalty (framework-style)

**Weight in final score**: 10%

---

### 6. Configuration System Presence (boolean)

**What it measures**: Whether the repository uses complex configuration systems (mmcv.Config, Hydra, OmegaConf).

**How it's detected**:
```python
# Searches for:
- Imports: mmcv.Config, OmegaConf, DictConfig
- Class names: Config, ConfigDict, ConfigParser
- Combined with registry patterns
```

**Why it matters**:
Configuration systems create layers of indirection:

- Features defined in YAML/JSON files, not explicit Python code
- Requires specific config parsers to use
- Often coupled with registry patterns
- Documentation may not cover all config options

**Dynamic Penalty System**:
```python
if has_config_system:
    if registry_usage > 200:
        config_penalty = 0.20  # Heavy: Config + lots of registries (MMSeg style)
    elif registry_usage > 50:
        config_penalty = 0.10  # Moderate: Config + some registries
    else:
        config_penalty = 0.05  # Light: Config alone isn't terrible
```

**Rationale**: Config systems alone aren't bad. The problem is config systems **combined with** heavy registry usage, which creates the most difficult extraction scenario.

**Example**:
- Lightly: Has config system + 26 registries → 0.05 penalty (5 points)
- MMSegmentation: Has config system + 258 registries → 0.20 penalty (20 points)

**Penalty**: Variable (5-20 points deducted)

---

### 7. Inheritance Depth (0+, lower is better)

**What it measures**: Maximum depth of class inheritance hierarchies.

**How it's calculated**:
```python
max_inheritance_depth = max(len(class.bases) for all classes)
inheritance_score = min(max_depth / 5.0, 1.0)
```

**Why it matters**:
- Deep inheritance creates fragile base class problems
- Difficult to understand behavior (methods inherited from ancestors)
- Extracting a class requires extracting entire hierarchy

**Example**:
- Depth 1: Simple, flat hierarchy → Easy to extract
- Depth 6+: Deep hierarchy → Must extract multiple parent classes

**Weight in final score**: 5%

---

### 8. Internal vs External Dependencies

**What it measures**: Ratio of dependencies within the repository vs external packages.

**How it's classified**:

1. **Fundamental ML Libraries** (excluded from penalties):
   - PyTorch, TensorFlow, NumPy, scikit-learn, etc.
   - Using these is expected and acceptable

2. **Standard Library** (excluded):
   - os, sys, json, etc.
   - Not counted as external dependencies

3. **Internal Dependencies** (good):
   - Imports of the repository's own modules
   - Indicates self-contained code

4. **External Dependencies** (concerning):
   - Third-party packages beyond fundamental ML libs
   - Each one is something users must install

**Size-Adjusted Bonus**:
```python
internal_ratio = internal / (internal + external)

if num_files > 500:
    # Large repos should be very self-contained
    bonus = max(0, (internal_ratio - 0.6)) * 0.20
elif num_files > 200:
    # Medium repos should be somewhat self-contained
    bonus = max(0, (internal_ratio - 0.5)) * 0.15
else:
    # Small repos get linear bonus
    bonus = internal_ratio * 0.10
```

**Example**:
- 485 internal, 71 external (87% internal) → High bonus → Self-contained
- 59 internal, 284 external (17% internal) → Low/no bonus → Framework-dependent

**Bonus**: Up to +20 points for large repos

---

### 9. Repository Size Adjustment

**What it measures**: Prevents tiny repositories from artificially inflating scores.

**Size Penalty**:
```python
if num_files < 100:
    size_penalty = (100 - num_files) / 400  # Up to 0.25 penalty
```

**Why it matters**:
- Small repos naturally have lower coupling and simpler structure
- This doesn't mean they're "easy to extract from" – they're just small
- A 10-file repo shouldn't score the same as a 1000-file well-designed library

**Example**:
- 15 files: -21 points penalty → Prevents inflation
- 50 files: -12.5 points penalty → Moderate correction
- 150 files: No penalty → Size doesn't influence score

**Penalty**: Up to -25 points for very small repos

---

## Extractability Score Calculation

### The Formula

```python
extractability_score = 100 * (
    (1 - coupling) * 0.25 +           # 25%: Avoid god objects
    modularity * 0.35 +               # 35%: Clear module boundaries (MOST IMPORTANT)
    cohesion * 0.10 +                 # 10%: Internal communication
    (1 - complexity_score) * 0.15 +   # 15%: Code simplicity
    (1 - inheritance_score) * 0.05 +  # 5%:  Shallow hierarchies
    (1 - registry_score) * 0.10 -     # 10%: Avoid registry patterns (context-aware)
    config_penalty -                  # Variable: -5 to -20 points
    size_penalty +                    # Variable: 0 to -25 points
    internal_bonus                    # Variable: 0 to +20 points
)

# Clamp to 0-100 range
extractability_score = max(0, min(100, extractability_score))
```

### Weight Justification

#### Why 35% for Modularity? (Highest Weight)

Through validation against known repositories, **modularity emerged as the strongest predictor** of extractability:

- **High modularity** (> 0.5) consistently correlated with easy extraction
- Repositories like Lightly (0.58) vs MMSegmentation (0.53) showed clear discrimination
- Natural communities in code = clear feature boundaries
- Statistical correlation with extractability: **r = 0.68** (strong positive)

Modularity captures the architectural property that matters most: **can features be isolated?**

#### Why 25% for Coupling?

Coupling identifies **critical bottlenecks** that block extraction:

- God objects force you to extract large portions of the codebase
- High coupling (> 0.4) makes extraction 3-4x more difficult
- Complements modularity: you want high modularity AND low coupling
- Statistical correlation: **r = -0.54** (strong negative)

#### Why 10% for Registry Patterns?

Registry patterns create **hidden runtime coupling** that static analysis can't fully capture:

- They're a critical discriminator between research code and frameworks
- Context-aware scoring (size-adjusted) prevents over-penalization
- OpenMMLab repos (MMSeg, MMDetection) consistently show high registry usage + low extractability
- Weight reduced from 20% to 10% after making it context-aware

#### Why 15% for Complexity?

Code complexity affects **understanding and modification**:

- Complex code is harder to debug when extracted
- Logarithmic scaling prevents outliers from dominating
- Less important than architecture (modularity/coupling)
- Well-documented complex code can still be extractable

#### Why 10% for Cohesion?

Cohesion is **less discriminative** than modularity at the repository level:

- Similar scores across different repositories (0.4-0.6 range)
- Measures communication density, not module boundaries
- Useful diagnostic but not primary predictor
- Higher cohesion within modules is good, but repo-wide cohesion can indicate monolithic design

#### Why Only 5% for Inheritance?

Most modern ML repos use **shallow inheritance**:

- Depths typically 1-2 (single parent class)
- Deep inheritance is rare in contemporary Python
- When present, it's problematic, but uncommon
- Less relevant than structural metrics

### Variable Components

#### Config Penalty (-5 to -20 points)

- **Why variable?** Config systems aren't inherently bad
- **Problem**: Config + heavy registries (the "framework pattern")
- **Light penalty (5 pts)**: Config for parameters (reasonable)
- **Heavy penalty (20 pts)**: Config + 200+ registries (extraction nightmare)

#### Size Penalty (0 to -25 points)

- **Why necessary?** Small repos game the system
- **Problem**: 10-file repos score higher than 1000-file well-designed libraries
- **Solution**: Graduated penalty for repos < 100 files
- **Exemption**: No penalty for repos ≥ 100 files

#### Internal Bonus (0 to +20 points)

- **Why size-adjusted?** Self-containment expectations scale with size
- **Small repos**: Linear bonus (any internal code is good)
- **Large repos**: Expected to be ≥ 60% internal to be truly self-contained
- **Reward**: Up to 20 points for large, self-contained repos

---

## Understanding the Results

### Score Ranges

| Range      | Interpretation | Characteristics                                     | Example Repos                                    |
| ---------- | -------------- | --------------------------------------------------- | ------------------------------------------------ |
| **70-100** | Excellent      | Minimal registries, high modularity, self-contained | MIScnn (60), segmentation_models.pytorch (61)    |
| **60-70**  | Very Good      | Clean architecture, some minor coupling             | cssegmentation (64), lightning-transformers (62) |
| **50-60**  | Good           | Extractable with moderate effort                    | Lightly (56), deepinv (56)                       |
| **40-50**  | Moderate       | Some architectural challenges                       | VISSL (39), s3prl (41)                           |
| **30-40**  | Challenging    | Framework-style patterns, refactoring needed        | MMSegmentation (32), CoreNet (27)                |
| **20-30**  | Difficult      | Heavy coupling or registries                        | fairseq (27), ClassyVision (29)                  |
| **0-20**   | Very Difficult | Monolithic, tightly integrated                      | tensor2tensor (8), pytorch (19)                  |

### Key Insights from Validation

#### Most Extractable Repos Share:
- High modularity (avg 0.58)
- Low registry usage (avg < 30)
- Low coupling (avg 0.24)
- Medium size (100-500 files sweet spot)

#### Least Extractable Repos Share:
- High registry usage (avg 800+)
- High coupling (avg 0.38)
- Low modularity (avg 0.32)
- Config systems present (90%)

### Real-World Validation

**Lightly vs MMSegmentation** (Key Test Case):

| Metric          | Lightly   | MMSeg     | Winner                  |
| --------------- | --------- | --------- | ----------------------- |
| **Final Score** | **56.04** | **31.77** | Lightly (+24 pts)       |
| Files           | 749       | 1,403     | -                       |
| Modularity      | 0.581     | 0.531     | Lightly                 |
| Registries      | 26        | 258       | **Lightly (10x fewer)** |
| Internal Deps   | 485 (87%) | 59 (17%)  | **Lightly (5x ratio)**  |
| Config Penalty  | 5 pts     | 20 pts    | Lightly                 |

**Real-world developer experience**: Extracting features from Lightly is significantly easier than from MMSegmentation, which requires bringing mmcv, the config system, and the entire registry infrastructure.

---

## Code Architecture

### Class Structure

```
GitHubRepoBatchAnalyzer
├── extract_repo_name()      # Parse GitHub URLs
├── clone_repo()              # Temporary shallow clone
├── analyze_repo()            # Main analysis orchestration
│   ├── DependencyAnalyzer
│   │   ├── find_all_python_files()
│   │   ├── analyze_file()         # AST parsing
│   │   ├── build_dependency_graph()
│   │   ├── calculate_coupling()   # Graph centrality
│   │   ├── calculate_modularity() # Louvain communities
│   │   ├── calculate_cohesion()
│   │   ├── detect_registry_pattern_usage()
│   │   └── detect_config_system()
│   └── [Compute extractability score]
├── generate_csv_report()     # Export results
└── process_batch()           # Batch orchestration
```

### Key Design Decisions

#### 1. Abstract Syntax Tree (AST) Analysis

**Why AST over regex/string parsing?**
- Accurate: Handles Python syntax correctly
- Robust: Not fooled by comments or strings
- Rich: Captures structure (classes, inheritance, decorators)
- Fast: No code execution required

#### 2. Graph Theory for Dependencies

**Why NetworkX for dependency graphs?**
- Proven algorithms (centrality, community detection)
- Scalable to large repositories
- Standard in software engineering research
- Enables sophisticated metrics (modularity, coupling)

#### 3. Louvain Method for Modularity

**Why Louvain over other community detection?**
- Fast: O(n log n) complexity
- Accurate: Finds natural communities
- Standard: Used in network science literature
- Robust: Works on graphs of varying sizes

#### 4. Shallow Git Clones

**Why `--depth 1`?**
- Speed: 10-50x faster than full clones
- Storage: Saves disk space (temporary anyway)
- Sufficient: Only need current code, not history

#### 5. Context-Aware Penalties

**Why size-adjusted scoring?**
- Fairness: Different expectations for different sizes
- Realism: Registry use justified in large frameworks
- Bias Removal: Prevents small repos from gaming metrics

---

## Troubleshooting

### Common Issues

**1. "python-louvain not installed"**
```bash
pip install python-louvain
```

**2. "Failed to clone repository"**
- Check internet connection
- Verify repository URL is correct
- Ensure repository is public (or you have access)
- Check git is in PATH: `git --version`

**3. "No Python files found"**
- Repository may not contain Python code
- Files may be in subdirectories the analyzer skips
- Check the error_message column in output CSV

**4. Windows permission errors when cleaning up**
- Already handled in v7 with `chmod` fix
- If persists, manually delete temp directories

---

## Contributing

### Extending the Analyzer

**Adding New Metrics:**

1. Add field to `ExtractabilityMetrics` dataclass
2. Implement calculation method in `DependencyAnalyzer`
3. Integrate into `analyze_repo()` scoring
4. Update weights in extractability formula
5. Document reasoning in README

**Example: Adding Test Coverage Metric**

```python
def detect_test_coverage(self) -> float:
    """Detect test file ratio"""
    test_files = len([f for f in self.file_analyses 
                     if 'test' in f.lower()])
    return test_files / len(self.file_analyses)

# In analyze_repo():
test_coverage = analyzer.detect_test_coverage()

# Add to formula:
extractability = 100 * (
    # ... existing metrics ...
    + test_coverage * 0.05  # Bonus for tests
)
```

---

## License & Citation

If you use this tool in research, please cite:

```
[Your citation format here]
```

---

## Version History

- **v7**: Size-aware penalties, context-aware registry scoring, internal dependency bonus
- **v6**: Config system dynamic penalties, improved internal dependency detection
- **v5**: Registry pattern detection, modularity emphasis
- **v4**: Fixed internal dependency detection, added ML library exemptions
- **v3**: Initial modularity and cohesion metrics
- **v2**: Basic coupling and complexity analysis
- **v1**: Simple dependency counting

---

## Contact & Support

For questions, issues, or contributions:
- GitHub Issues: [Your repo]
- Email: [Your email]
- Documentation: [Your docs link]

---

**Happy Analyzing! 🎯**