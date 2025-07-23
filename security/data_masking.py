"""
Data Masking for ChatBI platform.
Provides data anonymization and privacy protection for sensitive information.
"""

import re
import hashlib
import random
import string
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd

from utils.logger import get_logger
from utils.exceptions import DataProcessingException, ErrorCodes

logger = get_logger(__name__)


class MaskingType(Enum):
    """Types of data masking techniques."""
    FULL_MASK = "full_mask"
    PARTIAL_MASK = "partial_mask"
    HASH = "hash"
    ENCRYPT = "encrypt"
    SHUFFLE = "shuffle"
    SUBSTITUTE = "substitute"
    DATE_SHIFT = "date_shift"
    NULL_OUT = "null_out"
    FORMAT_PRESERVE = "format_preserve"


class SensitivityLevel(Enum):
    """Data sensitivity levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class DataMasker:
    """
    Comprehensive data masking system that provides various techniques
    for protecting sensitive data while maintaining analytical utility.
    """

    def __init__(self):
        """Initialize data masker with configuration."""

        # Masking rules based on column patterns
        self.column_rules = {
            # Personal Identifiable Information (PII)
            r'.*email.*': {
                'sensitivity': SensitivityLevel.CONFIDENTIAL,
                'masking_type': MaskingType.PARTIAL_MASK,
                'preserve_domain': True
            },
            r'.*phone.*': {
                'sensitivity': SensitivityLevel.CONFIDENTIAL,
                'masking_type': MaskingType.FORMAT_PRESERVE,
                'pattern': 'XXX-XXX-XXXX'
            },
            r'.*ssn.*|.*social.*': {
                'sensitivity': SensitivityLevel.RESTRICTED,
                'masking_type': MaskingType.HASH
            },
            r'.*credit.*card.*|.*cc.*': {
                'sensitivity': SensitivityLevel.RESTRICTED,
                'masking_type': MaskingType.PARTIAL_MASK,
                'show_last': 4
            },
            r'.*password.*': {
                'sensitivity': SensitivityLevel.RESTRICTED,
                'masking_type': MaskingType.FULL_MASK
            },
            r'.*address.*|.*addr.*': {
                'sensitivity': SensitivityLevel.CONFIDENTIAL,
                'masking_type': MaskingType.SUBSTITUTE
            },
            r'.*name.*': {
                'sensitivity': SensitivityLevel.CONFIDENTIAL,
                'masking_type': MaskingType.SUBSTITUTE,
                'maintain_format': True
            },
            r'.*salary.*|.*wage.*|.*income.*': {
                'sensitivity': SensitivityLevel.CONFIDENTIAL,
                'masking_type': MaskingType.SHUFFLE,
                'range_percent': 20
            },
            r'.*birth.*date.*|.*dob.*': {
                'sensitivity': SensitivityLevel.CONFIDENTIAL,
                'masking_type': MaskingType.DATE_SHIFT,
                'shift_range_days': 365
            }
        }

        # Substitute data for realistic masking
        self.substitute_data = {
            'first_names': [
                'John', 'Jane', 'Michael', 'Sarah', 'David', 'Emma', 'James', 'Emily',
                'Robert', 'Jessica', 'William', 'Ashley', 'Richard', 'Amanda', 'Thomas'
            ],
            'last_names': [
                'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
                'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez'
            ],
            'company_names': [
                'TechCorp', 'DataSystems', 'InnovateLLC', 'GlobalTech', 'FutureSoft',
                'SmartSolutions', 'CloudFirst', 'NextGen', 'ProTech', 'EliteData'
            ],
            'cities': [
                'Springfield', 'Franklin', 'Clinton', 'Madison', 'Georgetown',
                'Salem', 'Fairview', 'Bristol', 'Clayton', 'Riverside'
            ],
            'streets': [
                'Main St', 'Oak Ave', 'First St', 'Second St', 'Park Ave',
                'Elm St', 'Washington St', 'Maple Ave', 'Cedar St', 'Pine St'
            ]
        }

        # Format patterns for masked data
        self.format_patterns = {
            'phone': r'(\d{3})-(\d{3})-(\d{4})',
            'ssn': r'(\d{3})-(\d{2})-(\d{4})',
            'credit_card': r'(\d{4})\s?(\d{4})\s?(\d{4})\s?(\d{4})',
            'email': r'([^@]+)@([^.]+\..+)'
        }

        # Encryption key for consistent masking (in production, use proper key management)
        self.encryption_key = "ChatBI_Default_Key_2024"

    async def mask_dataframe(
            self,
            df: pd.DataFrame,
            user_permissions: Dict[str, Any],
            custom_rules: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Apply data masking to an entire DataFrame based on user permissions.

        Args:
            df: DataFrame to mask
            user_permissions: User's data access permissions
            custom_rules: Custom masking rules for specific columns

        Returns:
            Masked DataFrame
        """
        try:
            if df.empty:
                return df

            masked_df = df.copy()

            # Apply masking to each column
            for column in df.columns:
                column_rule = self._get_column_rule(column, custom_rules)

                if column_rule and self._should_mask_column(column, user_permissions, column_rule):
                    masked_df[column] = await self._mask_column(
                        df[column], column_rule, column
                    )

            logger.info(f"Applied data masking to {len(df.columns)} columns, {len(df)} rows")
            return masked_df

        except Exception as e:
            logger.error(f"Data masking failed: {e}")
            raise DataProcessingException(
                f"Failed to apply data masking: {str(e)}",
                ErrorCodes.DATA_TRANSFORMATION_ERROR
            )

    async def mask_query_results(
            self,
            results: List[Dict[str, Any]],
            user_permissions: Dict[str, Any],
            table_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply data masking to query results.

        Args:
            results: Query results to mask
            user_permissions: User's data access permissions
            table_metadata: Metadata about tables and columns

        Returns:
            Masked query results
        """
        try:
            if not results:
                return results

            # Convert to DataFrame for easier processing
            df = pd.DataFrame(results)
            masked_df = await self.mask_dataframe(df, user_permissions)

            # Convert back to list of dictionaries
            return masked_df.to_dict('records')

        except Exception as e:
            logger.error(f"Query result masking failed: {e}")
            return results  # Return original data if masking fails

    async def _mask_column(
            self,
            series: pd.Series,
            rule: Dict[str, Any],
            column_name: str
    ) -> pd.Series:
        """Apply masking to a specific column."""
        masking_type = MaskingType(rule['masking_type'])

        if masking_type == MaskingType.FULL_MASK:
            return self._full_mask(series)
        elif masking_type == MaskingType.PARTIAL_MASK:
            return self._partial_mask(series, rule)
        elif masking_type == MaskingType.HASH:
            return self._hash_mask(series)
        elif masking_type == MaskingType.SHUFFLE:
            return self._shuffle_mask(series, rule)
        elif masking_type == MaskingType.SUBSTITUTE:
            return self._substitute_mask(series, column_name, rule)
        elif masking_type == MaskingType.DATE_SHIFT:
            return self._date_shift_mask(series, rule)
        elif masking_type == MaskingType.FORMAT_PRESERVE:
            return self._format_preserve_mask(series, rule)
        elif masking_type == MaskingType.NULL_OUT:
            return self._null_out_mask(series)
        else:
            return series  # No masking applied

    def _full_mask(self, series: pd.Series) -> pd.Series:
        """Replace all values with masked characters."""
        return series.apply(lambda x: '*' * len(str(x)) if pd.notna(x) else x)

    def _partial_mask(self, series: pd.Series, rule: Dict[str, Any]) -> pd.Series:
        """Apply partial masking based on rule configuration."""

        def mask_value(value):
            if pd.isna(value):
                return value

            str_value = str(value)

            # Handle email masking
            if rule.get('preserve_domain') and '@' in str_value:
                return self._mask_email(str_value)

            # Handle credit card masking
            if rule.get('show_last'):
                show_last = rule['show_last']
                if len(str_value) > show_last:
                    return '*' * (len(str_value) - show_last) + str_value[-show_last:]

            # Default partial masking (show first and last characters)
            if len(str_value) <= 2:
                return '*' * len(str_value)
            elif len(str_value) <= 4:
                return str_value[0] + '*' * (len(str_value) - 2) + str_value[-1]
            else:
                return str_value[:2] + '*' * (len(str_value) - 4) + str_value[-2:]

        return series.apply(mask_value)

    def _hash_mask(self, series: pd.Series) -> pd.Series:
        """Apply hash-based masking for consistent anonymization."""

        def hash_value(value):
            if pd.isna(value):
                return value

            # Create consistent hash
            hash_input = f"{self.encryption_key}_{str(value)}"
            hash_object = hashlib.sha256(hash_input.encode())
            return hash_object.hexdigest()[:16]  # Use first 16 characters

        return series.apply(hash_value)

    def _shuffle_mask(self, series: pd.Series, rule: Dict[str, Any]) -> pd.Series:
        """Shuffle numeric values within a specified range."""

        def shuffle_value(value):
            if pd.isna(value) or not isinstance(value, (int, float)):
                return value

            range_percent = rule.get('range_percent', 10)
            variation = value * (range_percent / 100)

            # Add random variation
            random.seed(hash(str(value) + self.encryption_key) % (2 ** 32))
            new_value = value + random.uniform(-variation, variation)

            # Preserve data type
            if isinstance(value, int):
                return int(round(new_value))
            else:
                return round(new_value, 2)

        return series.apply(shuffle_value)

    def _substitute_mask(self, series: pd.Series, column_name: str, rule: Dict[str, Any]) -> pd.Series:
        """Replace values with realistic substitutes."""

        def substitute_value(value):
            if pd.isna(value):
                return value

            # Determine substitute category based on column name
            str_value = str(value)
            column_lower = column_name.lower()

            # Use consistent random seed for same value
            random.seed(hash(str_value + self.encryption_key) % (2 ** 32))

            if 'first' in column_lower or 'fname' in column_lower:
                return random.choice(self.substitute_data['first_names'])
            elif 'last' in column_lower or 'lname' in column_lower:
                return random.choice(self.substitute_data['last_names'])
            elif 'company' in column_lower or 'org' in column_lower:
                return random.choice(self.substitute_data['company_names'])
            elif 'city' in column_lower:
                return random.choice(self.substitute_data['cities'])
            elif 'street' in column_lower or 'address' in column_lower:
                number = random.randint(100, 9999)
                street = random.choice(self.substitute_data['streets'])
                return f"{number} {street}"
            else:
                # Generic substitution
                if rule.get('maintain_format'):
                    return self._maintain_format_substitute(str_value)
                else:
                    return f"MASKED_{random.randint(1000, 9999)}"

        return series.apply(substitute_value)

    def _date_shift_mask(self, series: pd.Series, rule: Dict[str, Any]) -> pd.Series:
        """Shift dates by a random amount within specified range."""

        def shift_date(value):
            if pd.isna(value):
                return value

            try:
                # Convert to datetime if not already
                if not isinstance(value, datetime):
                    date_value = pd.to_datetime(value)
                else:
                    date_value = value

                shift_range = rule.get('shift_range_days', 365)

                # Use consistent random seed
                random.seed(hash(str(value) + self.encryption_key) % (2 ** 32))
                shift_days = random.randint(-shift_range, shift_range)

                new_date = date_value + timedelta(days=shift_days)

                # Return in same format as input
                if isinstance(value, str):
                    return new_date.strftime('%Y-%m-%d')
                else:
                    return new_date

            except Exception:
                return value  # Return original if conversion fails

        return series.apply(shift_date)

    def _format_preserve_mask(self, series: pd.Series, rule: Dict[str, Any]) -> pd.Series:
        """Mask while preserving the original format."""

        def format_preserve(value):
            if pd.isna(value):
                return value

            str_value = str(value)
            pattern = rule.get('pattern', 'XXX-XXX-XXXX')

            # Replace digits with X, preserve other characters
            result = ''
            pattern_index = 0

            for char in str_value:
                if pattern_index < len(pattern):
                    if pattern[pattern_index] == 'X' and char.isdigit():
                        result += 'X'
                    elif pattern[pattern_index] == char:
                        result += char
                    else:
                        result += 'X' if char.isdigit() else char

                    if pattern[pattern_index] == 'X' or pattern[pattern_index] == char:
                        pattern_index += 1
                else:
                    result += 'X' if char.isdigit() else char

            return result

        return series.apply(format_preserve)

    def _null_out_mask(self, series: pd.Series) -> pd.Series:
        """Replace all values with NULL."""
        return pd.Series([None] * len(series), index=series.index)

    def _mask_email(self, email: str) -> str:
        """Mask email while preserving domain."""
        if '@' not in email:
            return email

        local, domain = email.split('@', 1)

        if len(local) <= 2:
            masked_local = '*' * len(local)
        else:
            masked_local = local[0] + '*' * (len(local) - 2) + local[-1]

        return f"{masked_local}@{domain}"

    def _maintain_format_substitute(self, value: str) -> str:
        """Create substitute that maintains the format of original."""
        result = ''

        for char in value:
            if char.isalpha():
                if char.isupper():
                    result += random.choice(string.ascii_uppercase)
                else:
                    result += random.choice(string.ascii_lowercase)
            elif char.isdigit():
                result += str(random.randint(0, 9))
            else:
                result += char

        return result

    def _get_column_rule(
            self,
            column_name: str,
            custom_rules: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Get masking rule for a column."""
        column_lower = column_name.lower()

        # Check custom rules first
        if custom_rules and column_name in custom_rules:
            return custom_rules[column_name]

        # Check pattern-based rules
        for pattern, rule in self.column_rules.items():
            if re.match(pattern, column_lower):
                return rule

        return None

    def _should_mask_column(
            self,
            column_name: str,
            user_permissions: Dict[str, Any],
            column_rule: Dict[str, Any]
    ) -> bool:
        """Determine if column should be masked for user."""
        sensitivity = SensitivityLevel(column_rule['sensitivity'])
        user_level = user_permissions.get('data_access_level', 'public')

        # Define access hierarchy
        access_levels = {
            'public': [SensitivityLevel.PUBLIC],
            'internal': [SensitivityLevel.PUBLIC, SensitivityLevel.INTERNAL],
            'confidential': [SensitivityLevel.PUBLIC, SensitivityLevel.INTERNAL, SensitivityLevel.CONFIDENTIAL],
            'restricted': [SensitivityLevel.PUBLIC, SensitivityLevel.INTERNAL, SensitivityLevel.CONFIDENTIAL,
                           SensitivityLevel.RESTRICTED]
        }

        allowed_levels = access_levels.get(user_level, [SensitivityLevel.PUBLIC])

        # Mask if user doesn't have sufficient access level
        return sensitivity not in allowed_levels

    async def get_masking_report(
            self,
            df: pd.DataFrame,
            user_permissions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a report of what would be masked."""
        try:
            report = {
                'total_columns': len(df.columns),
                'masked_columns': [],
                'unmasked_columns': [],
                'masking_techniques': {},
                'sensitivity_levels': {}
            }

            for column in df.columns:
                column_rule = self._get_column_rule(column, None)

                if column_rule and self._should_mask_column(column, user_permissions, column_rule):
                    masking_type = column_rule['masking_type']
                    sensitivity = column_rule['sensitivity']

                    report['masked_columns'].append({
                        'column': column,
                        'masking_type': masking_type,
                        'sensitivity': sensitivity
                    })

                    # Count techniques
                    report['masking_techniques'][masking_type] = report['masking_techniques'].get(masking_type, 0) + 1
                    report['sensitivity_levels'][sensitivity] = report['sensitivity_levels'].get(sensitivity, 0) + 1
                else:
                    report['unmasked_columns'].append(column)

            report['masking_percentage'] = (len(report['masked_columns']) / len(df.columns)) * 100

            return report

        except Exception as e:
            logger.error(f"Failed to generate masking report: {e}")
            return {'error': str(e)}

    def add_custom_rule(
            self,
            column_pattern: str,
            masking_type: MaskingType,
            sensitivity: SensitivityLevel,
            additional_params: Optional[Dict[str, Any]] = None
    ):
        """Add a custom masking rule."""
        rule = {
            'masking_type': masking_type,
            'sensitivity': sensitivity
        }

        if additional_params:
            rule.update(additional_params)

        self.column_rules[column_pattern] = rule
        logger.info(f"Added custom masking rule for pattern: {column_pattern}")

    def remove_custom_rule(self, column_pattern: str):
        """Remove a custom masking rule."""
        if column_pattern in self.column_rules:
            del self.column_rules[column_pattern]
            logger.info(f"Removed custom masking rule for pattern: {column_pattern}")

    def get_masking_configuration(self) -> Dict[str, Any]:
        """Get current masking configuration."""
        return {
            'column_rules': {
                pattern: {
                    'masking_type': rule['masking_type'],
                    'sensitivity': rule['sensitivity']
                }
                for pattern, rule in self.column_rules.items()
            },
            'total_rules': len(self.column_rules),
            'available_techniques': [mt.value for mt in MaskingType],
            'sensitivity_levels': [sl.value for sl in SensitivityLevel]
        }