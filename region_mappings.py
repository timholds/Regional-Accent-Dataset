"""
US Regional Accent Classification Mappings

This module provides mappings between US states and regional accent classifications
at three levels of granularity: coarse (5 regions), medium (8 regions), and fine (12 regions).
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


@dataclass
class RegionInfo:
    """Information about a dialect region"""
    name: str
    states: List[str]
    key_features: List[str]
    population_millions: int
    percentage: float


class CoarseRegion(Enum):
    """5-region coarse classification"""
    NORTHEAST = "Northeast"
    SOUTH = "South"
    MIDWEST = "Midwest"
    WEST = "West"
    TEXAS_SOUTHWEST = "Texas/Southwest"


class MediumRegion(Enum):
    """8-region medium classification (TIMIT-aligned)"""
    NEW_ENGLAND = "New England"  # dr1
    NEW_YORK = "New York Metropolitan"  # dr6
    MID_ATLANTIC = "Mid-Atlantic"  # dr3 partial
    SOUTH_ATLANTIC = "South Atlantic"  # dr5 partial
    DEEP_SOUTH = "Deep South"  # dr5 partial
    UPPER_MIDWEST = "Upper Midwest"  # dr2
    LOWER_MIDWEST = "Lower Midwest"  # dr3 partial
    WEST = "West"  # dr7


class FineRegion(Enum):
    """12-region fine classification"""
    EASTERN_NEW_ENGLAND = "Eastern New England"
    BOSTON_METRO = "Boston Metropolitan"
    NYC = "New York City"
    GREATER_NY = "Greater New York"
    PHILADELPHIA = "Philadelphia/South Jersey"
    MID_ATLANTIC = "Mid-Atlantic"
    UPPER_SOUTH = "Upper South"
    DEEP_SOUTH = "Deep South"
    FLORIDA = "Florida"
    GREAT_LAKES = "Great Lakes"
    MIDLAND = "Midland"
    WEST = "West"


# State abbreviations
STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
]


# Coarse classification (5 regions)
COARSE_MAPPINGS: Dict[str, CoarseRegion] = {
    # Northeast
    'ME': CoarseRegion.NORTHEAST, 'NH': CoarseRegion.NORTHEAST,
    'VT': CoarseRegion.NORTHEAST, 'MA': CoarseRegion.NORTHEAST,
    'RI': CoarseRegion.NORTHEAST, 'CT': CoarseRegion.NORTHEAST,
    'NY': CoarseRegion.NORTHEAST, 'NJ': CoarseRegion.NORTHEAST,
    'PA': CoarseRegion.NORTHEAST,
    
    # South
    'DE': CoarseRegion.SOUTH, 'MD': CoarseRegion.SOUTH,
    'DC': CoarseRegion.SOUTH, 'VA': CoarseRegion.SOUTH,
    'WV': CoarseRegion.SOUTH, 'KY': CoarseRegion.SOUTH,
    'TN': CoarseRegion.SOUTH, 'NC': CoarseRegion.SOUTH,
    'SC': CoarseRegion.SOUTH, 'GA': CoarseRegion.SOUTH,
    'FL': CoarseRegion.SOUTH, 'AL': CoarseRegion.SOUTH,
    'MS': CoarseRegion.SOUTH, 'LA': CoarseRegion.SOUTH,
    'AR': CoarseRegion.SOUTH,
    
    # Midwest
    'OH': CoarseRegion.MIDWEST, 'IN': CoarseRegion.MIDWEST,
    'IL': CoarseRegion.MIDWEST, 'MI': CoarseRegion.MIDWEST,
    'WI': CoarseRegion.MIDWEST, 'MN': CoarseRegion.MIDWEST,
    'IA': CoarseRegion.MIDWEST, 'MO': CoarseRegion.MIDWEST,
    'ND': CoarseRegion.MIDWEST, 'SD': CoarseRegion.MIDWEST,
    'NE': CoarseRegion.MIDWEST, 'KS': CoarseRegion.MIDWEST,
    
    # West
    'MT': CoarseRegion.WEST, 'ID': CoarseRegion.WEST,
    'WY': CoarseRegion.WEST, 'CO': CoarseRegion.WEST,
    'NM': CoarseRegion.WEST, 'AZ': CoarseRegion.WEST,
    'UT': CoarseRegion.WEST, 'NV': CoarseRegion.WEST,
    'WA': CoarseRegion.WEST, 'OR': CoarseRegion.WEST,
    'CA': CoarseRegion.WEST, 'AK': CoarseRegion.WEST,
    'HI': CoarseRegion.WEST,
    
    # Texas/Southwest
    'TX': CoarseRegion.TEXAS_SOUTHWEST, 'OK': CoarseRegion.TEXAS_SOUTHWEST,
}


# Medium classification (8 regions)
MEDIUM_MAPPINGS: Dict[str, MediumRegion] = {
    # New England
    'ME': MediumRegion.NEW_ENGLAND, 'NH': MediumRegion.NEW_ENGLAND,
    'VT': MediumRegion.NEW_ENGLAND, 'MA': MediumRegion.NEW_ENGLAND,
    'RI': MediumRegion.NEW_ENGLAND, 'CT': MediumRegion.NEW_ENGLAND,
    
    # New York Metropolitan
    'NY': MediumRegion.NEW_YORK,  # Will need sub-region handling
    'NJ': MediumRegion.NEW_YORK,  # Northern NJ only
    
    # Mid-Atlantic
    'PA': MediumRegion.MID_ATLANTIC, 'DE': MediumRegion.MID_ATLANTIC,
    'MD': MediumRegion.MID_ATLANTIC, 'DC': MediumRegion.MID_ATLANTIC,
    
    # South Atlantic
    'VA': MediumRegion.SOUTH_ATLANTIC, 'NC': MediumRegion.SOUTH_ATLANTIC,
    'SC': MediumRegion.SOUTH_ATLANTIC, 'GA': MediumRegion.SOUTH_ATLANTIC,
    'FL': MediumRegion.SOUTH_ATLANTIC,
    
    # Deep South
    'AL': MediumRegion.DEEP_SOUTH, 'MS': MediumRegion.DEEP_SOUTH,
    'LA': MediumRegion.DEEP_SOUTH, 'AR': MediumRegion.DEEP_SOUTH,
    'TN': MediumRegion.DEEP_SOUTH, 'KY': MediumRegion.DEEP_SOUTH,
    
    # Upper Midwest
    'WI': MediumRegion.UPPER_MIDWEST, 'MI': MediumRegion.UPPER_MIDWEST,
    'MN': MediumRegion.UPPER_MIDWEST, 'ND': MediumRegion.UPPER_MIDWEST,
    'SD': MediumRegion.UPPER_MIDWEST, 'IL': MediumRegion.UPPER_MIDWEST,  # Northern IL
    
    # Lower Midwest
    'OH': MediumRegion.LOWER_MIDWEST, 'IN': MediumRegion.LOWER_MIDWEST,
    'MO': MediumRegion.LOWER_MIDWEST, 'IA': MediumRegion.LOWER_MIDWEST,
    'NE': MediumRegion.LOWER_MIDWEST, 'KS': MediumRegion.LOWER_MIDWEST,
    
    # West (including TX, OK)
    'MT': MediumRegion.WEST, 'ID': MediumRegion.WEST,
    'WY': MediumRegion.WEST, 'CO': MediumRegion.WEST,
    'NM': MediumRegion.WEST, 'AZ': MediumRegion.WEST,
    'UT': MediumRegion.WEST, 'NV': MediumRegion.WEST,
    'WA': MediumRegion.WEST, 'OR': MediumRegion.WEST,
    'CA': MediumRegion.WEST, 'AK': MediumRegion.WEST,
    'HI': MediumRegion.WEST, 'TX': MediumRegion.WEST,
    'OK': MediumRegion.WEST, 'WV': MediumRegion.WEST,
}


# Fine classification (12 regions)
FINE_MAPPINGS: Dict[str, FineRegion] = {
    # Eastern New England
    'ME': FineRegion.EASTERN_NEW_ENGLAND, 'NH': FineRegion.EASTERN_NEW_ENGLAND,
    'VT': FineRegion.EASTERN_NEW_ENGLAND,
    
    # Boston Metro (special handling needed)
    'MA': FineRegion.BOSTON_METRO,  # Eastern MA only
    
    # NYC (special handling needed)
    'NY': FineRegion.NYC,  # NYC boroughs only
    
    # Greater NY
    'CT': FineRegion.GREATER_NY,  # Southwest CT
    'NJ': FineRegion.GREATER_NY,  # Northern NJ
    
    # Philadelphia/South Jersey
    'PA': FineRegion.PHILADELPHIA,  # SE PA only
    # 'NJ': FineRegion.PHILADELPHIA,  # Southern NJ
    'DE': FineRegion.PHILADELPHIA,  # Northern DE
    
    # Mid-Atlantic
    'MD': FineRegion.MID_ATLANTIC, 'DC': FineRegion.MID_ATLANTIC,
    'WV': FineRegion.MID_ATLANTIC,
    
    # Upper South
    'VA': FineRegion.UPPER_SOUTH, 'NC': FineRegion.UPPER_SOUTH,
    'KY': FineRegion.UPPER_SOUTH, 'TN': FineRegion.UPPER_SOUTH,  # Eastern TN
    
    # Deep South
    'GA': FineRegion.DEEP_SOUTH, 'AL': FineRegion.DEEP_SOUTH,
    'MS': FineRegion.DEEP_SOUTH, 'SC': FineRegion.DEEP_SOUTH,
    'AR': FineRegion.DEEP_SOUTH, 'LA': FineRegion.DEEP_SOUTH,
    
    # Florida
    'FL': FineRegion.FLORIDA,
    
    # Great Lakes
    'WI': FineRegion.GREAT_LAKES, 'MI': FineRegion.GREAT_LAKES,
    'OH': FineRegion.GREAT_LAKES,  # Northern OH
    'IN': FineRegion.GREAT_LAKES,  # Northern IN
    'IL': FineRegion.GREAT_LAKES,  # Northern IL
    'MN': FineRegion.GREAT_LAKES,
    
    # Midland
    'MO': FineRegion.MIDLAND, 'IA': FineRegion.MIDLAND,
    'NE': FineRegion.MIDLAND, 'KS': FineRegion.MIDLAND,
    'OK': FineRegion.MIDLAND,
    
    # West
    'MT': FineRegion.WEST, 'ID': FineRegion.WEST,
    'WY': FineRegion.WEST, 'CO': FineRegion.WEST,
    'NM': FineRegion.WEST, 'AZ': FineRegion.WEST,
    'UT': FineRegion.WEST, 'NV': FineRegion.WEST,
    'WA': FineRegion.WEST, 'OR': FineRegion.WEST,
    'CA': FineRegion.WEST, 'AK': FineRegion.WEST,
    'HI': FineRegion.WEST, 'TX': FineRegion.WEST,
    'ND': FineRegion.WEST, 'SD': FineRegion.WEST,
    'RI': FineRegion.EASTERN_NEW_ENGLAND,
}


# TIMIT dialect region mapping
TIMIT_REGIONS = {
    'dr1': 'New England',
    'dr2': 'Northern',
    'dr3': 'North Midland',
    'dr4': 'South Midland',
    'dr5': 'Southern',
    'dr6': 'New York City',
    'dr7': 'Western',
    'dr8': 'Army Brat (moved around)'
}


def get_region_for_state(state: str, classification: str = 'medium') -> str:
    """
    Get the dialect region for a given state abbreviation.
    
    Args:
        state: Two-letter state abbreviation (e.g., 'NY', 'CA')
        classification: 'coarse', 'medium', or 'fine'
    
    Returns:
        Region name as string
    """
    state = state.upper()
    
    if classification == 'coarse':
        return COARSE_MAPPINGS.get(state, CoarseRegion.WEST).value
    elif classification == 'medium':
        return MEDIUM_MAPPINGS.get(state, MediumRegion.WEST).value
    elif classification == 'fine':
        return FINE_MAPPINGS.get(state, FineRegion.WEST).value
    else:
        raise ValueError(f"Unknown classification: {classification}")


def get_states_for_region(region: str, classification: str = 'medium') -> List[str]:
    """
    Get all states in a given dialect region.
    
    Args:
        region: Region name
        classification: 'coarse', 'medium', or 'fine'
    
    Returns:
        List of state abbreviations
    """
    states = []
    
    if classification == 'coarse':
        mapping = COARSE_MAPPINGS
        for state, reg in mapping.items():
            if reg.value == region:
                states.append(state)
    elif classification == 'medium':
        mapping = MEDIUM_MAPPINGS
        for state, reg in mapping.items():
            if reg.value == region:
                states.append(state)
    elif classification == 'fine':
        mapping = FINE_MAPPINGS
        for state, reg in mapping.items():
            if reg.value == region:
                states.append(state)
    
    return sorted(states)


def get_timit_region(state: str) -> Tuple[str, str]:
    """
    Get the TIMIT dialect region code and name for a state.
    
    Args:
        state: Two-letter state abbreviation
    
    Returns:
        Tuple of (TIMIT code, region name)
    """
    # Simplified mapping to TIMIT regions
    timit_mapping = {
        'dr1': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT'],
        'dr2': ['WI', 'MI', 'MN', 'ND', 'SD'],
        'dr3': ['PA', 'OH', 'IN', 'IL'],
        'dr4': ['WV', 'KY', 'TN', 'MO', 'KS'],
        'dr5': ['VA', 'NC', 'SC', 'GA', 'FL', 'AL', 'MS', 'LA', 'AR'],
        'dr6': ['NY', 'NJ'],
        'dr7': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'TX', 'OK', 'NE', 'IA', 'AK', 'HI'],
        'dr8': []  # Army brat - no specific states
    }
    
    state = state.upper()
    for code, states in timit_mapping.items():
        if state in states:
            return code, TIMIT_REGIONS[code]
    
    # Default to Western if not found
    return 'dr7', TIMIT_REGIONS['dr7']


if __name__ == "__main__":
    # Example usage
    print("Example State-to-Region Mappings:")
    print("-" * 50)
    
    test_states = ['MA', 'NY', 'GA', 'TX', 'CA', 'IL']
    
    for state in test_states:
        print(f"\n{state}:")
        print(f"  Coarse: {get_region_for_state(state, 'coarse')}")
        print(f"  Medium: {get_region_for_state(state, 'medium')}")
        print(f"  Fine: {get_region_for_state(state, 'fine')}")
        timit_code, timit_name = get_timit_region(state)
        print(f"  TIMIT: {timit_code} - {timit_name}")
    
    print("\n\nStates in each Medium region:")
    print("-" * 50)
    for region in MediumRegion:
        states = get_states_for_region(region.value, 'medium')
        print(f"{region.value}: {', '.join(states)}")