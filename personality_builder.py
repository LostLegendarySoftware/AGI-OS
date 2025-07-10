import random
from typing import List, Dict, Any
from dataclasses import dataclass, field
from core.ariel import ArielAgent, ArielSignals
from core.exceptions import PersonalityBuildError
from core.logging import logger

@dataclass
class Personality:
    traits: Dict[str, float] = field(default_factory=dict)
    interests: List[str] = field(default_factory=list)
    values: Dict[str, float] = field(default_factory=dict)
    background: Dict[str, Any] = field(default_factory=dict)

class PersonalityBuilder:
    def __init__(self, agent: ArielAgent):
        self.agent = agent
        self.personality = Personality()

    async def build_personality(self) -> Personality:
        try:
            await self._generate_traits()
            await self._generate_interests()
            await self._generate_values()
            await self._generate_background()
            await ArielSignals.personality_built.emit(personality=self.personality)
            return self.personality
        except Exception as e:
            logger.error(f"Failed to build personality: {str(e)}")
            raise PersonalityBuildError(f"Failed to build personality: {str(e)}")

    async def _generate_traits(self):
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        self.personality.traits = {trait: random.uniform(0, 1) for trait in traits}

    async def _generate_interests(self):
        possible_interests = ["science", "art", "technology", "philosophy", "nature", "history", "music"]
        self.personality.interests = random.sample(possible_interests, random.randint(2, 5))

    async def _generate_values(self):
        values = ["honesty", "curiosity", "efficiency", "creativity", "empathy"]
        self.personality.values = {value: random.uniform(0, 1) for value in values}

    async def _generate_background(self):
        self.personality.background = {
            "origin": random.choice(["research lab", "tech startup", "university project", "government initiative"]),
            "purpose": random.choice(["general assistance", "scientific research", "creative collaboration", "problem-solving"]),
            "age": random.randint(1, 5),  # in years
        }

    async def adjust_personality(self, aspect: str, value: Any):
        if aspect in self.personality.traits:
            self.personality.traits[aspect] = max(0, min(1, value))
        elif aspect in self.personality.values:
            self.personality.values[aspect] = max(0, min(1, value))
        elif aspect == "interests":
            if isinstance(value, list):
                self.personality.interests = value
            else:
                raise PersonalityBuildError("Interests must be a list")
        elif aspect in self.personality.background:
            self.personality.background[aspect] = value
        else:
            raise PersonalityBuildError(f"Unknown personality aspect: {aspect}")

        await ArielSignals.personality_adjusted.emit(aspect=aspect, value=value)