"""
Natural Language Instruction Parser for Embodied Navigation
============================================================

Converts text instructions into structured navigation goals.

Two modes:
  1. VLM mode (Ollama available): Parse complex instructions via LLM
  2. Regex mode (lightweight): Parse simple commands via pattern matching
     → For Go2 deployment where Ollama may not be available

Supported instruction types:
  - Object goal:  "find a toilet" / "go to the bed" / "navigate to a chair"
  - Room goal:    "go to the kitchen" / "navigate to bedroom" / "go to room 3"
  - Landmark:     "go near the fridge" / "go to where the couch is"
  - Descriptive:  "find the room with the red couch" (VLM mode only)
  - Stop:         "stop" / "halt" / "stay here"

Usage:
    from perception.instruction_parser import InstructionParser

    parser = InstructionParser(mode='regex')     # Lightweight
    parser = InstructionParser(mode='vlm', ollama_host='http://localhost:11434')

    goal = parser.parse("go to the kitchen")
    # → NavGoal(type='room', target='kitchen', raw='go to the kitchen')

    goal = parser.parse("find a toilet")
    # → NavGoal(type='object', target='toilet', raw='find a toilet')
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class NavGoal:
    """Structured navigation goal from parsed instruction."""
    type: str           # 'object', 'room', 'landmark', 'stop', 'unknown'
    target: str         # e.g. 'toilet', 'kitchen', 'room_3'
    raw: str            # Original instruction text
    confidence: float = 1.0
    attributes: Dict = field(default_factory=dict)  # e.g. {'color': 'red'}


# ================================================================
# Room and Object Vocabularies
# ================================================================

ROOM_TYPES = {
    'bedroom', 'bathroom', 'kitchen', 'living room', 'living_room',
    'dining room', 'dining_room', 'hallway', 'corridor', 'office',
    'study', 'garage', 'laundry', 'closet', 'basement', 'attic',
    'entrance', 'foyer', 'lounge', 'den', 'pantry', 'balcony',
    'patio', 'nursery', 'guest room', 'master bedroom',
}

# Aliases → canonical room name
ROOM_ALIASES = {
    'bath': 'bathroom', 'restroom': 'bathroom', 'washroom': 'bathroom',
    'toilet room': 'bathroom', 'loo': 'bathroom', 'wc': 'bathroom',
    'bed room': 'bedroom', 'sleeping room': 'bedroom',
    'living': 'living_room', 'family room': 'living_room',
    'sitting room': 'living_room', 'tv room': 'living_room',
    'dining': 'dining_room', 'eating area': 'dining_room',
    'hall': 'hallway', 'passage': 'hallway', 'entry': 'hallway',
    'workspace': 'office', 'home office': 'office',
    'storage': 'closet', 'utility': 'laundry',
}

# HM3D ObjectNav categories
OBJECT_CATEGORIES = {
    'chair', 'bed', 'plant', 'toilet', 'sofa', 'couch',
    'tv', 'tv_monitor', 'television', 'monitor',
    'table', 'desk', 'refrigerator', 'fridge', 'sink',
    'oven', 'microwave', 'lamp', 'clock', 'vase',
    'book', 'laptop', 'bottle',
}

OBJECT_ALIASES = {
    'potted plant': 'plant', 'houseplant': 'plant',
    'television': 'tv_monitor', 'tv': 'tv_monitor', 'monitor': 'tv_monitor',
    'couch': 'sofa', 'settee': 'sofa', 'loveseat': 'sofa',
    'fridge': 'refrigerator', 'icebox': 'refrigerator',
    'dining table': 'table', 'coffee table': 'table',
}


# ================================================================
# Regex-based Parser (lightweight, for Go2 deployment)
# ================================================================

class RegexParser:
    """
    Pattern-matching instruction parser.
    No ML model needed — runs on any hardware.
    """

    # Patterns ordered by specificity
    PATTERNS = [
        # Stop commands
        (r'\b(stop|halt|stay|freeze|wait)\b', 'stop', None),

        # Room with number: "go to room 3", "room number 5"
        (r'(?:go\s+to|navigate\s+to|find|head\s+to)\s+room\s*(?:number\s*)?(\d+)',
         'room', lambda m: f'room_{m.group(1)}'),

        # Named room: "go to the kitchen"
        (r'(?:go\s+to|navigate\s+to|find|head\s+to|take\s+me\s+to|move\s+to)\s+(?:the\s+)?(.+?)(?:\s+room)?$',
         '_check_room_or_object', None),

        # Landmark: "go near the fridge", "go towards the couch"
        (r'(?:go\s+near|go\s+towards|go\s+by|head\s+towards|move\s+near)\s+(?:the\s+)?(.+?)$',
         '_check_object', None),

        # Object with article: "find a toilet", "find the bed"
        (r'(?:find|locate|search\s+for|look\s+for|where\s+is)\s+(?:a\s+|the\s+|an\s+)?(.+?)$',
         '_check_object', None),

        # Simple noun: "kitchen", "toilet", "bedroom"
        (r'^(.+?)$', '_check_any', None),
    ]

    def parse(self, text: str) -> NavGoal:
        text = text.lower().strip()
        text = re.sub(r'[.!?]+$', '', text).strip()

        # Try each pattern
        for pattern, goal_type, extractor in self.PATTERNS:
            m = re.search(pattern, text)
            if m:
                if goal_type == 'stop':
                    return NavGoal(type='stop', target='stop', raw=text)

                if goal_type == 'room':
                    target = extractor(m) if extractor else m.group(1)
                    return NavGoal(type='room', target=target, raw=text)

                if goal_type == '_check_room_or_object':
                    target_text = m.group(1).strip()
                    return self._classify_target(target_text, text)

                if goal_type == '_check_object':
                    target_text = m.group(1).strip()
                    return self._classify_target(target_text, text)

                if goal_type == '_check_any':
                    target_text = m.group(1).strip()
                    return self._classify_target(target_text, text)

        return NavGoal(type='unknown', target=text, raw=text, confidence=0.0)

    def _classify_target(self, target_text: str, raw: str) -> NavGoal:
        """Determine if target is a room, object, or unknown."""
        t = target_text.lower().strip()
        # Strip leading articles
        t = re.sub(r'^(a|an|the|some|any)\s+', '', t).strip()

        # Check room aliases first
        if t in ROOM_ALIASES:
            return NavGoal(type='room', target=ROOM_ALIASES[t], raw=raw)

        # Check room types (require full word match, not substring)
        t_words = set(t.split())
        for room in ROOM_TYPES:
            room_words = set(room.split())
            # Full match: all room words present in target
            if room_words.issubset(t_words) or t == room.replace(' ', '_'):
                canonical = room.replace(' ', '_')
                return NavGoal(type='room', target=canonical, raw=raw)

        # Check object aliases
        if t in OBJECT_ALIASES:
            return NavGoal(type='object', target=OBJECT_ALIASES[t], raw=raw)

        # Check object categories (exact match)
        if t in OBJECT_CATEGORIES:
            return NavGoal(type='object', target=t, raw=raw)

        # Fuzzy: contains a room-ish word?
        room_hints = ['room', 'area', 'space', 'zone']
        if any(h in t for h in room_hints):
            return NavGoal(type='room', target=t.replace(' ', '_'), raw=raw,
                           confidence=0.5)

        return NavGoal(type='unknown', target=t, raw=raw, confidence=0.3)


# ================================================================
# VLM-based Parser (Ollama, for powerful hardware)
# ================================================================

class VLMParser:
    """
    Uses local VLM to parse complex natural language instructions.
    Handles ambiguous, descriptive, or multi-step commands.
    """

    def __init__(self, ollama_host='http://localhost:11434', model='llava:7b'):
        self.host = ollama_host
        self.model = model
        self._available = None
        self._regex_fallback = RegexParser()

    @property
    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import requests
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def parse(self, text: str, scene_context: str = None) -> NavGoal:
        """
        Parse instruction using VLM.

        Args:
            text: Natural language instruction
            scene_context: Optional scene graph description for context
        """
        # Try regex first for simple commands (faster)
        simple = self._regex_fallback.parse(text)
        if simple.confidence >= 0.8 and simple.type != 'unknown':
            return simple

        if not self.available:
            return simple

        try:
            import requests

            context_str = ""
            if scene_context:
                context_str = f"\nKnown environment: {scene_context}\n"

            prompt = f"""Parse this navigation instruction into a structured goal.
{context_str}
Instruction: "{text}"

Reply with EXACTLY one line in this format:
TYPE:TARGET

Where TYPE is one of: object, room, landmark, stop
And TARGET is the specific thing to navigate to.

Examples:
- "find a toilet" → object:toilet
- "go to the kitchen" → room:kitchen
- "find the room with the TV" → room:living_room
- "go near the fridge" → landmark:refrigerator
- "stop here" → stop:stop

Your answer:"""

            r = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 30},
                },
                timeout=10,
            )

            if r.status_code == 200:
                response = r.json().get('response', '').strip()
                return self._parse_vlm_response(response, text)

        except Exception:
            pass

        return simple

    def _parse_vlm_response(self, response: str, raw: str) -> NavGoal:
        """Parse VLM's TYPE:TARGET response."""
        # Clean up response
        line = response.strip().split('\n')[0].strip()
        line = line.lstrip('- ').strip()

        if ':' in line:
            parts = line.split(':', 1)
            goal_type = parts[0].strip().lower()
            target = parts[1].strip().lower().replace(' ', '_')

            if goal_type in ('object', 'room', 'landmark', 'stop'):
                return NavGoal(type=goal_type, target=target, raw=raw)

        # Fallback to regex
        return self._regex_fallback.parse(raw)


# ================================================================
# Main Parser (auto-selects mode)
# ================================================================

class InstructionParser:
    """
    Unified instruction parser.

    mode='auto': Try VLM first, fall back to regex
    mode='vlm':  VLM only (with regex fallback on error)
    mode='regex': Regex only (lightweight, for Go2)
    """

    def __init__(self, mode='auto', ollama_host='http://localhost:11434',
                 vlm_model='llava:7b'):
        self.mode = mode

        if mode in ('auto', 'vlm'):
            self._vlm = VLMParser(ollama_host, vlm_model)
        else:
            self._vlm = None

        self._regex = RegexParser()

    def parse(self, text: str, scene_context: str = None) -> NavGoal:
        """Parse a natural language navigation instruction."""
        if self.mode == 'regex':
            return self._regex.parse(text)

        if self._vlm and self._vlm.available:
            return self._vlm.parse(text, scene_context)

        return self._regex.parse(text)

    def parse_batch(self, instructions: List[str]) -> List[NavGoal]:
        """Parse multiple instructions."""
        return [self.parse(t) for t in instructions]


# ================================================================
# Integration with Navigation System
# ================================================================

def goal_to_action(goal: NavGoal, scene_graph=None, annotated_map=None):
    """
    Convert a NavGoal into a navigation target position.

    Uses scene graph and/or annotated map to resolve goals to positions.

    Returns:
        dict with 'target_pos' (x, y map coords) and 'strategy'
        or None if goal cannot be resolved
    """
    if goal.type == 'stop':
        return {'action': 'stop', 'target_pos': None, 'strategy': 'stop'}

    if goal.type == 'room':
        # Try annotated map — match by room TYPE, not exact label
        # "kitchen" should match "kitchen_1", "kitchen_2", etc.
        if annotated_map:
            rooms = annotated_map.get('rooms', {})

            # First try exact label match
            if goal.target in rooms:
                room_data = rooms[goal.target]
                return {
                    'target_pos': room_data['center'],
                    'strategy': 'navigate_to_room',
                    'room': goal.target,
                }

            # Then match by room type
            for label, room_data in rooms.items():
                if room_data.get('type') == goal.target:
                    return {
                        'target_pos': room_data['center'],
                        'strategy': 'navigate_to_room',
                        'room': label,
                    }

            # Fuzzy: label starts with target
            for label, room_data in rooms.items():
                if label.startswith(goal.target):
                    return {
                        'target_pos': room_data['center'],
                        'strategy': 'navigate_to_room',
                        'room': label,
                    }

        # Try scene graph (discovered rooms)
        if scene_graph:
            for node in scene_graph.nodes.values():
                if node.node_type == 'room' and node.label == goal.target:
                    return {
                        'target_pos': node.position,
                        'strategy': 'navigate_to_room',
                        'room': goal.target,
                    }

        return {
            'target_pos': None,
            'strategy': 'explore_for_room',
            'room': goal.target,
        }

    if goal.type in ('object', 'landmark'):
        # Check annotated map objects first
        if annotated_map:
            for obj in annotated_map.get('objects', []):
                obj_label = obj.get('label', '').lower()
                if obj_label == goal.target or goal.target in obj_label:
                    return {
                        'target_pos': obj['position'],
                        'strategy': 'navigate_to_object' if goal.type == 'object' else 'navigate_near',
                        'object': obj_label,
                        'room': obj.get('room'),
                    }

        # Check scene graph
        if scene_graph:
            for node in scene_graph.nodes.values():
                if node.node_type == 'object':
                    if node.label == goal.target or goal.target in node.label:
                        return {
                            'target_pos': node.position,
                            'strategy': 'navigate_to_object' if goal.type == 'object' else 'navigate_near',
                            'object': node.label,
                        }

        return {
            'target_pos': None,
            'strategy': f'explore_for_{goal.type}',
            goal.type: goal.target,
        }

    return None


# ================================================================
# Quick Test
# ================================================================

if __name__ == '__main__':
    parser = InstructionParser(mode='regex')

    test_instructions = [
        "find a toilet",
        "go to the kitchen",
        "navigate to bedroom",
        "go to room 3",
        "find the bed",
        "where is the TV",
        "stop",
        "take me to the living room",
        "go near the fridge",
        "find a chair",
        "head to the bathroom",
        "go to the office",
        "look for a plant",
        "search for the sofa",
    ]

    print("Regex Parser Tests:")
    print("-" * 60)
    for instr in test_instructions:
        goal = parser.parse(instr)
        print(f"  '{instr}'")
        print(f"    → type={goal.type}, target={goal.target}, conf={goal.confidence}")
    print()