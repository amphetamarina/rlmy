"""
Purpose: Find-and-replace with fuzzy matching, ported from OpenCode's edit.ts.
Usage: new_content = replace(content, old_text, new_text)

Source: https://github.com/sst/opencode/blob/main/packages/opencode/src/tool/edit.ts

Core insight: The 'old' string IS the context. Include enough surrounding
lines in 'old' to make the match unique.

Strategies (tried in order until one finds a unique match):
1. Simple - exact match
2. Line-trimmed - trim whitespace per line
3. Block-anchor - Levenshtein fuzzy match by first/last line anchors
4. Whitespace-normalized - collapse all whitespace to single space
5. Indentation-flexible - ignore common indentation
6. Escape-normalized - handle \n, \t, etc.
7. Trimmed-boundary - try stripped version
8. Context-aware - first/last anchors with 50% middle similarity
"""

from typing import Iterator, Callable


# Similarity thresholds for block-anchor strategy (from OpenCode)
SINGLE_CANDIDATE_SIMILARITY_THRESHOLD = 0.6
MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD = 0.8


# Type alias: Replacer takes (content, find_string) and yields potential matches
Replacer = Callable[[str, str], Iterator[str]]


def replace(content: str, old: str, new: str) -> str:
    """
    Replace old with new in content using fuzzy matching strategies.
    
    Tries each strategy in order until one finds a unique match.
    
    Args:
        content: The file content
        old: Text to find (can be multi-line, include context for uniqueness)
        new: Replacement text
    
    Returns:
        Content with old replaced by new
    
    Raises:
        ValueError: If old == new, not found, or multiple matches
    """
    if old == new:
        raise ValueError("old and new must be different")
    
    # Special case: empty old on empty content (file creation)
    if old == "" and content == "":
        return new
    
    # Special case: empty old on non-empty content is ambiguous
    if old == "":
        raise ValueError("empty old string on non-empty content is ambiguous")
    
    strategies: list[Replacer] = [
        simple_replacer,
        line_trimmed_replacer,
        block_anchor_replacer,
        whitespace_normalized_replacer,
        indentation_flexible_replacer,
        escape_normalized_replacer,
        trimmed_boundary_replacer,
        context_aware_replacer,
    ]
    
    found_multiple = False  # Track if we found matches but they were ambiguous
    
    for strategy in strategies:
        for match in strategy(content, old):
            index = content.find(match)
            if index == -1:
                continue
            
            # Check uniqueness: is there only one occurrence?
            last_index = content.rfind(match)
            if index != last_index:
                found_multiple = True
                continue  # Multiple matches, try next strategy
            
            # Single unique match — do the replacement
            return content[:index] + new + content[index + len(match):]
    
    if found_multiple:
        raise ValueError(
            f"Found multiple matches for old. Provide more surrounding lines "
            f"in old to identify the correct match. old: {old[:100]}..."
        )
    raise ValueError(f"old not found in content. Provide more context or check for typos. old: {old[:100]}...")


# =============================================================================
# HELPER: LEVENSHTEIN DISTANCE
# =============================================================================

def levenshtein(a: str, b: str) -> int:
    """
    Calculate edit distance between two strings.
    Used by block_anchor_replacer for fuzzy matching.
    
    Ported from OpenCode L185-201.
    """
    if not a or not b:
        return max(len(a), len(b))
    
    matrix = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        matrix[i][0] = i
    for j in range(len(b) + 1):
        matrix[0][j] = j
    
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            matrix[i][j] = min(
                matrix[i-1][j] + 1,      # deletion
                matrix[i][j-1] + 1,      # insertion
                matrix[i-1][j-1] + cost  # substitution
            )
    return matrix[len(a)][len(b)]


# =============================================================================
# MATCHING STRATEGIES
# =============================================================================

def simple_replacer(content: str, find: str) -> Iterator[str]:
    """
    Strategy 1: Exact match.
    Just yields the find string as-is.
    
    Ported from OpenCode L203-205.
    """
    yield find


def line_trimmed_replacer(content: str, find: str) -> Iterator[str]:
    """
    Strategy 2: Match lines with trimmed whitespace.
    
    Compares lines after stripping leading/trailing whitespace,
    but returns the original content (preserving actual whitespace).
    
    Ported from OpenCode L207-245.
    """
    content_lines = content.split("\n")
    find_lines = find.split("\n")
    
    # Remove trailing empty line if present (common when find ends with \n)
    if find_lines and find_lines[-1] == "":
        find_lines = find_lines[:-1]
    
    if not find_lines:
        return
    
    # Slide through content looking for matching block
    for i in range(len(content_lines) - len(find_lines) + 1):
        matches = True
        for j, find_line in enumerate(find_lines):
            if content_lines[i + j].strip() != find_line.strip():
                matches = False
                break
        
        if matches:
            # Return the actual content (preserving original whitespace)
            matched_lines = content_lines[i:i + len(find_lines)]
            yield "\n".join(matched_lines)


def block_anchor_replacer(content: str, find: str) -> Iterator[str]:
    """
    Strategy 3: Match by first/last line anchors with Levenshtein similarity.
    
    Uses first and last lines as exact-match "anchors", then calculates
    similarity of middle lines using Levenshtein distance.
    
    Requires at least 3 lines in find string.
    
    Ported from OpenCode L247-380.
    """
    content_lines = content.split("\n")
    find_lines = find.split("\n")
    
    # Need at least 3 lines for meaningful anchors
    if len(find_lines) < 3:
        return
    
    # Remove trailing empty line if present
    if find_lines and find_lines[-1] == "":
        find_lines = find_lines[:-1]
    
    if len(find_lines) < 3:
        return
    
    first_line_find = find_lines[0].strip()
    last_line_find = find_lines[-1].strip()
    find_block_size = len(find_lines)
    
    # Collect all candidate positions where both anchors match
    candidates: list[tuple[int, int]] = []  # (start_line, end_line)
    for i in range(len(content_lines)):
        if content_lines[i].strip() != first_line_find:
            continue
        
        # Look for matching last line after this first line
        for j in range(i + 2, len(content_lines)):
            if content_lines[j].strip() == last_line_find:
                candidates.append((i, j))
                break  # Only match first occurrence of last line
    
    if not candidates:
        return
    
    # Handle single candidate (relaxed threshold)
    if len(candidates) == 1:
        start_line, end_line = candidates[0]
        actual_block_size = end_line - start_line + 1
        
        similarity = 0.0
        lines_to_check = min(find_block_size - 2, actual_block_size - 2)
        
        if lines_to_check > 0:
            for k in range(1, min(find_block_size - 1, actual_block_size - 1)):
                content_line = content_lines[start_line + k].strip()
                find_line = find_lines[k].strip()
                max_len = max(len(content_line), len(find_line))
                if max_len == 0:
                    continue
                distance = levenshtein(content_line, find_line)
                similarity += (1 - distance / max_len) / lines_to_check
                
                if similarity >= SINGLE_CANDIDATE_SIMILARITY_THRESHOLD:
                    break
        else:
            similarity = 1.0
        
        if similarity >= SINGLE_CANDIDATE_SIMILARITY_THRESHOLD:
            matched = content_lines[start_line:end_line + 1]
            yield "\n".join(matched)
        return
    
    # Handle multiple candidates (stricter threshold)
    best_match: tuple[int, int] | None = None
    max_similarity = -1.0
    
    for start_line, end_line in candidates:
        actual_block_size = end_line - start_line + 1
        
        similarity = 0.0
        lines_to_check = min(find_block_size - 2, actual_block_size - 2)
        
        if lines_to_check > 0:
            for k in range(1, min(find_block_size - 1, actual_block_size - 1)):
                content_line = content_lines[start_line + k].strip()
                find_line = find_lines[k].strip()
                max_len = max(len(content_line), len(find_line))
                if max_len == 0:
                    continue
                distance = levenshtein(content_line, find_line)
                similarity += 1 - distance / max_len
            similarity /= lines_to_check
        else:
            similarity = 1.0
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = (start_line, end_line)
    
    if max_similarity >= MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD and best_match:
        start_line, end_line = best_match
        matched = content_lines[start_line:end_line + 1]
        yield "\n".join(matched)


def whitespace_normalized_replacer(content: str, find: str) -> Iterator[str]:
    """
    Strategy 4: Normalize whitespace to single spaces.
    
    Collapses all whitespace (spaces, tabs, newlines) to single spaces,
    then looks for matches.
    
    Ported from OpenCode L382-424.
    """
    def normalize(text: str) -> str:
        return " ".join(text.split())
    
    normalized_find = normalize(find)
    if not normalized_find:
        return
    
    # Single line matches
    for line in content.split("\n"):
        if normalize(line) == normalized_find:
            yield line
    
    # Multi-line matches
    find_lines = find.split("\n")
    if len(find_lines) > 1:
        content_lines = content.split("\n")
        for i in range(len(content_lines) - len(find_lines) + 1):
            block = content_lines[i:i + len(find_lines)]
            if normalize("\n".join(block)) == normalized_find:
                yield "\n".join(block)


def indentation_flexible_replacer(content: str, find: str) -> Iterator[str]:
    """
    Strategy 5: Match ignoring indentation differences.
    
    Removes the common leading indentation from both content block
    and find string, then compares.
    
    Ported from OpenCode L426-452.
    """
    def remove_indent(text: str) -> str:
        lines = text.split("\n")
        non_empty = [ln for ln in lines if ln.strip()]
        if not non_empty:
            return text
        
        # Find minimum indentation
        min_indent = min(len(ln) - len(ln.lstrip()) for ln in non_empty)
        
        # Remove that indentation from all lines
        return "\n".join(
            ln[min_indent:] if len(ln) > min_indent else ln.lstrip()
            for ln in lines
        )
    
    normalized_find = remove_indent(find)
    if not normalized_find.strip():
        return
    
    content_lines = content.split("\n")
    find_lines = find.split("\n")
    
    # Remove trailing empty line if present
    if find_lines and find_lines[-1] == "":
        find_lines = find_lines[:-1]
    
    if not find_lines:
        return
    
    for i in range(len(content_lines) - len(find_lines) + 1):
        block = "\n".join(content_lines[i:i + len(find_lines)])
        if remove_indent(block) == normalized_find:
            yield block


def escape_normalized_replacer(content: str, find: str) -> Iterator[str]:
    """
    Strategy 6: Handle escape sequences in find string.
    
    Converts escape sequences like \\n, \\t, \\r to actual characters.
    Useful when LLM outputs "line1\\nline2" instead of actual newlines.
    
    Ported from OpenCode L454-501.
    """
    def unescape(s: str) -> str:
        """Convert escape sequences to actual characters."""
        replacements = {
            '\\n': '\n',
            '\\t': '\t',
            '\\r': '\r',
            "\\'": "'",
            '\\"': '"',
            '\\`': '`',
            '\\\\': '\\',
            '\\$': '$',
        }
        result = s
        for escaped, unescaped in replacements.items():
            result = result.replace(escaped, unescaped)
        return result
    
    unescaped_find = unescape(find)
    
    # If nothing changed, no point in trying
    if unescaped_find == find:
        return
    
    # Try direct match with unescaped find string
    if unescaped_find in content:
        yield unescaped_find
    
    # Also try finding blocks where unescaped content matches
    content_lines = content.split("\n")
    find_lines = unescaped_find.split("\n")
    
    for i in range(len(content_lines) - len(find_lines) + 1):
        block = "\n".join(content_lines[i:i + len(find_lines)])
        if unescape(block) == unescaped_find:
            yield block


def trimmed_boundary_replacer(content: str, find: str) -> Iterator[str]:
    """
    Strategy 7: Try stripped version of find string.
    
    Useful when find has leading/trailing whitespace or newlines
    that don't exist in content.
    
    Ported from OpenCode L517-541.
    """
    trimmed = find.strip()
    
    if trimmed == find:
        return  # Already trimmed, no point in trying again
    
    if not trimmed:
        return
    
    # Direct match of trimmed version
    if trimmed in content:
        yield trimmed
    
    # Also try block matching with trimmed content
    content_lines = content.split("\n")
    find_lines = find.split("\n")
    
    # Remove empty lines from edges
    while find_lines and find_lines[0].strip() == "":
        find_lines = find_lines[1:]
    while find_lines and find_lines[-1].strip() == "":
        find_lines = find_lines[:-1]
    
    if not find_lines:
        return
    
    for i in range(len(content_lines) - len(find_lines) + 1):
        block = content_lines[i:i + len(find_lines)]
        if "\n".join(ln.strip() for ln in block) == "\n".join(ln.strip() for ln in find_lines):
            yield "\n".join(block)


def context_aware_replacer(content: str, find: str) -> Iterator[str]:
    """
    Strategy 8: Match using first/last lines as context anchors.
    
    Similar to block_anchor but simpler — requires 50% of middle lines
    to match exactly (after trim), no Levenshtein.
    
    Ported from OpenCode L543-599.
    """
    find_lines = find.split("\n")
    
    # Need at least 3 lines for meaningful context
    if len(find_lines) < 3:
        return
    
    # Remove trailing empty line if present
    if find_lines and find_lines[-1] == "":
        find_lines = find_lines[:-1]
    
    if len(find_lines) < 3:
        return
    
    content_lines = content.split("\n")
    
    first_line = find_lines[0].strip()
    last_line = find_lines[-1].strip()
    
    # Find blocks that start and end with context anchors
    for i in range(len(content_lines)):
        if content_lines[i].strip() != first_line:
            continue
        
        # Look for matching last line
        for j in range(i + 2, len(content_lines)):
            if content_lines[j].strip() != last_line:
                continue
            
            # Found potential context block
            block_lines = content_lines[i:j + 1]
            
            # Check if block has same number of lines
            if len(block_lines) != len(find_lines):
                break
            
            # Check middle line similarity (50% threshold)
            matching_lines = 0
            total_non_empty = 0
            
            for k in range(1, len(block_lines) - 1):
                block_line = block_lines[k].strip()
                find_line = find_lines[k].strip()
                
                if block_line or find_line:
                    total_non_empty += 1
                    if block_line == find_line:
                        matching_lines += 1
            
            if total_non_empty == 0 or matching_lines / total_non_empty >= 0.5:
                yield "\n".join(block_lines)
                return  # Only match first occurrence
            
            break  # Only check first matching last line

