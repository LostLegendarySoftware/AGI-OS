import asyncio
import json
import os
import random
import re
import string
from typing import List, Dict, Any

import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm

from ariel.core import ariel
from ariel.exceptions import ArielAgentGenerationError
from ariel.signals import ArielSignals
from ariel.models import ArielModel
from ariel.utils.file_utils import load_jsonl, save_json

import logging

logger = logging.getLogger(__name__)

class AgentGenerator:
    def __init__(self, output_dir: str, seed_tasks_path: str):
        self.output_dir = output_dir
        self.seed_tasks_path = seed_tasks_path
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        self.model = ArielModel()  # Initialize Ariel's model

    async def encode_prompt(self, prompt_agents: List[Dict[str, Any]]) -> str:
        """Encode multiple prompt instructions into a single string."""
        try:
            with open("./prompt.txt", "r") as f:
                prompt = f.read() + "\n"
        except FileNotFoundError:
            raise ArielAgentGenerationError("Prompt file not found")

        for idx, task_dict in enumerate(prompt_agents):
            name, goal = task_dict["name"], task_dict["goal"]
            if not goal:
                raise ArielAgentGenerationError("Empty goal in prompt agents")
            prompt += f"###\n{idx + 1}. Name: {name}\n{idx + 1}. Goal:\n{goal}\n"
        prompt += f"###\n{idx + 2}. Name:"
        return prompt

    async def post_process_ariel_response(self, num_prompt_agents: int, response: str) -> List[Dict[str, str]]:
        raw_instructions = f"{num_prompt_agents+1}. Name:" + response
        raw_instructions = re.split("###", raw_instructions)
        agents = []
        for idx, inst in enumerate(raw_instructions):
            idx += num_prompt_agents + 1
            splitted_data = re.split(f"{idx}\.\s+(Name|Goal):", inst)
            if len(splitted_data) != 5:
                continue
            name, goal = splitted_data[2].strip(), splitted_data[4].strip()
            if len(goal.split()) <= 3 or len(goal.split()) > 150:
                continue
            if await self.is_blacklisted(goal) or goal[0] in string.punctuation or not goal[0].isascii():
                continue
            agents.append({"name": name, "goal": goal})
        return agents

    @staticmethod
    async def is_blacklisted(goal: str) -> bool:
        blacklist = [
            "kill", "harm", "discriminate", "racist", "figure", "plot", "chart",
            "image", "images", "graph", "graphs", "picture", "pictures", "file",
            "files", "draw", "plot", "go to", "video", "audio", "flowchart", "diagram"
        ]
        return any(re.search(r"\b{0}\b".format(word), goal, re.IGNORECASE) for word in blacklist)

    @ariel.service
    async def generate_agents_data(
        self,
        num_agents_to_generate: int = 50,
        num_prompt_agents: int = 8,
        temperature: float = 1.0,
        top_p: float = 1.0
    ) -> List[Dict[str, Any]]:
        logger.info("Starting agent data generation")
        ArielSignals.agent_generation_started.emit(num_agents=num_agents_to_generate)

        seed_tasks = await load_jsonl(self.seed_tasks_path)
        seed_agent_data = [{"name": t["name"], "goal": t["goal"], "task_id": t["task_id"]} for t in seed_tasks]
        logger.info(f"Loaded {len(seed_agent_data)} human-written seed agents")

        os.makedirs(self.output_dir, exist_ok=True)
        machine_agent_data = await self.load_machine_agent_data()

        all_goals = list(set([d["goal"] for d in seed_agent_data + machine_agent_data]))
        all_instruction_tokens = [self.scorer._tokenizer.tokenize(role) for role in all_goals]

        progress_bar = tqdm(total=num_agents_to_generate)
        progress_bar.update(len(machine_agent_data))

        while len(machine_agent_data) < num_agents_to_generate:
            prompt_agents = random.sample(seed_agent_data, num_prompt_agents)
            prompt = await self.encode_prompt(prompt_agents)

            try:
                response = await self.model.generate(
                    prompt,
                    max_tokens=3072,
                    temperature=temperature,
                    top_p=top_p,
                    stop=["\n60", "60."]
                )
            except Exception as e:
                logger.error(f"Error in Ariel model generation: {str(e)}")
                ArielSignals.agent_generation_error.emit(error=str(e))
                raise ArielAgentGenerationError(f"Ariel model error: {str(e)}")

            agent_data = await self.post_process_ariel_response(num_prompt_agents, response)

            for agent_data_entry in agent_data:
                new_agent_tokens = self.scorer._tokenizer.tokenize(agent_data_entry["goal"])
                rouge_scores = await self.compute_rouge_scores(new_agent_tokens, all_instruction_tokens)
                
                max_score = max(rouge_scores)
                if max_score > 0.40:
                    continue

                agent_data_entry["max_similarity_score"] = max_score
                agent_data_entry["seed_tasks"] = [task["task_id"] for task in prompt_agents]
                machine_agent_data.append(agent_data_entry)
                all_goals.append(agent_data_entry["goal"])
                all_instruction_tokens.append(new_agent_tokens)
                progress_bar.update(1)

            await save_json(machine_agent_data, os.path.join(self.output_dir, "self-gen.json"))

        ArielSignals.agent_generation_completed.emit(num_agents=len(machine_agent_data))
        return machine_agent_data

    async def load_machine_agent_data(self) -> List[Dict[str, Any]]:
        machine_data_path = os.path.join(self.output_dir, "self-gen-batch1.json")
        if os.path.exists(machine_data_path):
            machine_agent_data = await load_jsonl(machine_data_path)
            logger.info(f"Loaded {len(machine_agent_data)} machine-generated agents")
        else:
            machine_agent_data = []
        return machine_agent_data

    async def compute_rouge_scores(self, new_agent_tokens: List[str], all_instruction_tokens: List[List[str]]) -> List[float]:
        rouge_scores = [
            self.scorer._score_lcs(new_agent_tokens, tokens).fmeasure
            for tokens in all_instruction_tokens
        ]
        return rouge_scores

@ariel.route('/generate_agents', methods=['POST'])
async def generate_agents_route(request):
    try:
        data = await request.json()
        generator = AgentGenerator(
            output_dir=data.get('output_dir', './'),
            seed_tasks_path=data.get('seed_tasks_path', './new_seed_tasks.jsonl')
        )
        agents = await generator.generate_agents_data(
            num_agents_to_generate=data.get('num_agents_to_generate', 50),
            num_prompt_agents=data.get('num_prompt_agents', 8),
            temperature=data.get('temperature', 1.0),
            top_p=data.get('top_p', 1.0)
        )
        return {'status': 'success', 'agents': agents}
    except ArielAgentGenerationError as e:
        logger.error(f"Agent generation error: {str(e)}")
        return {'status': 'error', 'message': str(e)}, 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {'status': 'error', 'message': 'An unexpected error occurred'}, 500