"""
Data normalization utilities for ATM project.

This module provides standardized normalization functions for various data types
used throughout the project, ensuring consistency across all components.

Standardization Rules:
- Stock codes: {code}.{market} format (6-digit code, uppercase market)
- Dates: YYYY-MM-DD format for display, YYYYMMDD for storage
- Index codes: Same as stock codes
- Industry codes: Standard format
- And more...
"""

import re
import pandas as pd
from datetime import datetime
from typing import Optional, Union


# ============================================================================
# Stock Code Normalization
# ============================================================================

def normalize_stock_code(code: str) -> str:
    """
    Normalize stock code to standard format: {code}.{market}
    
    Standard format:
    - code: 6-digit number (e.g., '000001', '600000')
    - market: Uppercase exchange code (SH, SZ, BJ)
    
    Supported input formats:
    - '000001.SZ' or '000001.sz' -> '000001.SZ'
    - '600000.SH' or '600000.sh' -> '600000.SH'
    - 'sh.000001' or 'SZ.000001' -> '000001.SH' or '000001.SZ'
    - '000001' -> '000001.SZ' (default to SZ if starts with 0/3)
    - '600000' -> '600000.SH' (default to SH if starts with 6)
    - 'SH000300' -> '000300.SH'
    - 'SZ399001' -> '399001.SZ'
    
    Args:
        code: Stock code in various formats.
    
    Returns:
        Normalized stock code in standard format: {code}.{market}
        Returns empty string if input is invalid.
    
    Examples:
        >>> normalize_stock_code('000001.SZ')
        '000001.SZ'
        >>> normalize_stock_code('600000.sh')
        '600000.SH'
        >>> normalize_stock_code('sh.000001')
        '000001.SH'
        >>> normalize_stock_code('000001')
        '000001.SZ'
        >>> normalize_stock_code('600000')
        '600000.SH'
        >>> normalize_stock_code('SH000300')
        '000300.SH'
    """
    if not code:
        return ""
    
    code_str = str(code).strip()
    
    # Handle empty string
    if not code_str:
        return ""
    
    # Handle format: SH000300, SZ399001, BJ430001
    if len(code_str) >= 2 and code_str[:2].upper() in ['SH', 'SZ', 'BJ']:
        market = code_str[:2].upper()
        code_part = code_str[2:].strip()
        if code_part and code_part.isdigit():
            # Pad to 6 digits if needed
            code_part = code_part.zfill(6)
            return f"{code_part}.{market}"
    
    # Handle format with dot: 000001.SZ, sh.000001, etc.
    if '.' in code_str:
        parts = code_str.split('.')
        if len(parts) == 2:
            part1, part2 = parts[0].strip(), parts[1].strip()
            
            # Case 1: code.market (e.g., '000001.SZ', '600000.sh')
            if part1.isdigit() and part2.upper() in ['SH', 'SZ', 'BJ']:
                code_part = part1.zfill(6)
                market = part2.upper()
                return f"{code_part}.{market}"
            
            # Case 2: market.code (e.g., 'sh.000001', 'SZ.600000')
            if part1.upper() in ['SH', 'SZ', 'BJ'] and part2.isdigit():
                market = part1.upper()
                code_part = part2.zfill(6)
                return f"{code_part}.{market}"
    
    # Handle pure numeric code (e.g., '000001', '600000')
    if code_str.isdigit():
        code_part = code_str.zfill(6)
        # Determine market based on first digit
        if code_part.startswith('6'):
            return f"{code_part}.SH"
        elif code_part.startswith(('0', '3')):
            return f"{code_part}.SZ"
        elif code_part.startswith('4'):
            return f"{code_part}.BJ"
        else:
            # Default to SZ for unknown patterns
            return f"{code_part}.SZ"
    
    # If cannot parse, try to extract digits and market
    # Extract all digits
    digits = ''.join(re.findall(r'\d+', code_str))
    if digits:
        code_part = digits.zfill(6)
        # Try to find market indicator
        code_upper = code_str.upper()
        if 'SH' in code_upper or code_part.startswith('6'):
            return f"{code_part}.SH"
        elif 'SZ' in code_upper or code_part.startswith(('0', '3')):
            return f"{code_part}.SZ"
        elif 'BJ' in code_upper or code_part.startswith('4'):
            return f"{code_part}.BJ"
        else:
            return f"{code_part}.SZ"  # Default
    
    # Return as-is if cannot parse (may be index code or other format)
    return code_str.upper()


def normalize_stock_codes(codes: list[str]) -> list[str]:
    """
    Normalize a list of stock codes to standard format.
    
    Args:
        codes: List of stock codes in various formats.
    
    Returns:
        List of normalized stock codes in standard format.
    """
    return [normalize_stock_code(code) for code in codes if code]


def extract_code_and_market(normalized_code: str) -> tuple[str, str]:
    """
    Extract code and market from normalized stock code.
    
    Args:
        normalized_code: Stock code in standard format: {code}.{market}
    
    Returns:
        Tuple of (code, market), e.g., ('000001', 'SZ')
    
    Raises:
        ValueError: If code format is invalid.
    """
    if '.' not in normalized_code:
        raise ValueError(f"Invalid stock code format: {normalized_code}")
    
    parts = normalized_code.split('.')
    if len(parts) != 2:
        raise ValueError(f"Invalid stock code format: {normalized_code}")
    
    code, market = parts[0], parts[1].upper()
    
    if not code.isdigit() or market not in ['SH', 'SZ', 'BJ']:
        raise ValueError(f"Invalid stock code format: {normalized_code}")
    
    return code, market


def is_valid_stock_code(code: str) -> bool:
    """
    Check if a stock code is in valid standard format.
    
    Args:
        code: Stock code to validate.
    
    Returns:
        True if code is in valid standard format, False otherwise.
    """
    if not code or '.' not in code:
        return False
    
    try:
        code_part, market = extract_code_and_market(code)
        return len(code_part) == 6 and market in ['SH', 'SZ', 'BJ']
    except ValueError:
        return False


# ============================================================================
# Index Code Normalization
# ============================================================================

def normalize_index_code(code: str) -> str:
    """
    Normalize index code to standard format: {code}.{market}
    
    Index codes follow the same format as stock codes.
    
    Args:
        code: Index code in various formats.
    
    Returns:
        Normalized index code in standard format: {code}.{market}
    
    Examples:
        >>> normalize_index_code('000300.SH')
        '000300.SH'
        >>> normalize_index_code('CSI300')
        'CSI300'  # Returns as-is if cannot parse
    """
    # Try stock code normalization first
    normalized = normalize_stock_code(code)
    
    # If normalization failed (returned as-is), it might be an index code
    # Return as uppercase for consistency
    if normalized == code:
        return code.upper()
    
    return normalized


# ============================================================================
# Date Normalization
# ============================================================================

def normalize_date(date: Union[str, datetime]) -> str:
    """
    Normalize date to standard format: YYYY-MM-DD (for display).
    
    Args:
        date: Date in various formats (string or datetime object).
    
    Returns:
        Normalized date string in YYYY-MM-DD format.
    
    Examples:
        >>> normalize_date('2025-01-02')
        '2025-01-02'
        >>> normalize_date('20250102')
        '2025-01-02'
        >>> normalize_date(datetime(2025, 1, 2))
        '2025-01-02'
    """
    if isinstance(date, datetime):
        return date.strftime('%Y-%m-%d')
    
    if not date:
        return ""
    
    date_str = str(date).strip()
    
    # Handle YYYYMMDD format
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    # Handle YYYY-MM-DD format
    if len(date_str) == 10 and date_str.count('-') == 2:
        return date_str
    
    # Try to parse with pandas
    try:
        dt = pd.to_datetime(date_str)
        return dt.strftime('%Y-%m-%d')
    except Exception:
        # Return as-is if cannot parse
        return date_str


def normalize_date_for_storage(date: Union[str, datetime]) -> str:
    """
    Normalize date to storage format: YYYYMMDD (for database/Qlib).
    
    Args:
        date: Date in various formats (string or datetime object).
    
    Returns:
        Normalized date string in YYYYMMDD format.
    
    Examples:
        >>> normalize_date_for_storage('2025-01-02')
        '20250102'
        >>> normalize_date_for_storage('20250102')
        '20250102'
        >>> normalize_date_for_storage(datetime(2025, 1, 2))
        '20250102'
    """
    if isinstance(date, datetime):
        return date.strftime('%Y%m%d')
    
    if not date:
        return ""
    
    date_str = str(date).strip()
    
    # Handle YYYYMMDD format (already correct)
    if len(date_str) == 8 and date_str.isdigit():
        return date_str
    
    # Handle YYYY-MM-DD format
    if len(date_str) == 10 and date_str.count('-') == 2:
        return date_str.replace('-', '')
    
    # Try to parse with pandas
    try:
        dt = pd.to_datetime(date_str)
        return dt.strftime('%Y%m%d')
    except Exception:
        # Return as-is if cannot parse
        return date_str


# ============================================================================
# Industry Code Normalization
# ============================================================================

def normalize_industry_code(code: str) -> str:
    """
    Normalize industry code to standard format.
    
    Industry codes are typically in format like '850531.SI' (申万行业代码).
    
    Args:
        code: Industry code in various formats.
    
    Returns:
        Normalized industry code in standard format.
    
    Examples:
        >>> normalize_industry_code('850531.SI')
        '850531.SI'
        >>> normalize_industry_code('850531.si')
        '850531.SI'
    """
    if not code:
        return ""
    
    code_str = str(code).strip()
    
    # If has dot, normalize market suffix
    if '.' in code_str:
        parts = code_str.split('.')
        if len(parts) == 2:
            return f"{parts[0]}.{parts[1].upper()}"
    
    return code_str.upper()


# ============================================================================
# Qlib Directory Name Normalization
# ============================================================================

def normalize_qlib_directory_name(code: str) -> str:
    """
    Normalize code for Qlib directory name (lowercase for compatibility).
    
    Qlib uses lowercase for feature directories (e.g., 000001.sz),
    but we standardize to uppercase format (e.g., 000001.SZ) internally.
    This function converts to lowercase for Qlib directory compatibility.
    
    Args:
        code: Stock code in various formats.
    
    Returns:
        Normalized directory name (lowercase for Qlib compatibility).
    
    Examples:
        >>> normalize_qlib_directory_name('000001.SZ')
        '000001.sz'
        >>> normalize_qlib_directory_name('600000.SH')
        '600000.sh'
    """
    normalized = normalize_stock_code(code)
    if normalized:
        # Qlib uses lowercase for directory names
        return normalized.lower()
    return code.lower() if code else ""


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# Aliases for backward compatibility
normalize_ts_code = normalize_stock_code
convert_ts_code_to_qlib_format = normalize_stock_code

