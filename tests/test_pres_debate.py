from typing import Dict, List
import pytest
import json
import os
import tempfile

from openpyxl.styles.builtins import output

from docetl import Optimizer
from docetl.runner import DSLRunner
from docetl.utils import load_config
import yaml
from docetl.api import (
    Pipeline,
    Dataset,
    MapOp,
    UnnestOp,
    PipelineOutput,
    PipelineStep,
)

# Sample configuration for the test
# SAMPLE_CONFIG = """
# datasets:
#   debates:
#     type: file
#     path: "example_data/debates/data.json"
#
# system_prompt:
#   dataset_description: a collection of transcripts of presidential debates
#   persona: a political analyst
#
# default_model: gpt-4o-mini
#
# operations:
#   - name: extract_themes_and_viewpoints
#     type: map
#     output:
#       schema:
#         themes: "list[{fallacy: str, reasoning: str}]"
#     prompt: |
#       summarize all instances of logical fallacies that happened during the last 40 years of presidential debates for {{ input.title }} on {{ input.date }}:
#
#       {{ input.content }}
#
#       Extract the main topics that led to logical fallacies in this debate and the explanations that led to such logical fallacies by the candidates.
#       Return a list of fallacies and corresponding explanations in the following format:
#       [
#         {
#           "fallacy": "Topic 1",
#           "reasoning": "Candidate A's claim ... why is it a fallacy?"
#         },
#         {
#           "fallacy": "Topic 2",
#           "reasoning": "Candidate B's claim ... why is it a fallacy?"
#         },
#         ...
#       ]
#
# pipeline:
#   steps:
#     - name: debate_analysis
#       input: debates
#       operations:
#         - extract_themes_and_viewpoints
#
#   output:
#     type: file
#     path: "categorical_analysis_optimized.json"
#     intermediate_dir: "checkpoints"
# """
#
# @pytest.fixture
# def config_file():
#     with tempfile.NamedTemporaryFile(
#         mode="w+", suffix=".yaml", delete=False
#     ) as temp_file:
#         temp_file.write(SAMPLE_CONFIG)
#         temp_file.flush()
#         yield temp_file.name
#     os.unlink(temp_file.name)
#
#
# def test_debate_pipeline(config_file):
#     # Update the config with the correct sample data path
#     runner = DSLRunner.from_yaml(config_file)
#     optimizer = Optimizer(
#         runner=runner,
#         model="gpt-4o-mini",
#         timeout=30
#     )
#
#     # Run optimization
#     total_cost = optimizer.optimize()
#     print("total_cost",total_cost)
#
#     total_cost = runner.load_run_save()
#
#     print("total_cost", total_cost)


import pytest
import json
import tempfile
import os
from docetl.api import (
    Pipeline,
    Dataset,
    MapOp,
    ReduceOp,
    ParallelMapOp,
    FilterOp,
    PipelineStep,
    PipelineOutput,
    ResolveOp,
    EquijoinOp,
)
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def default_model():
    return "gpt-4o-mini"


@pytest.fixture
def max_threads():
    return 4


@pytest.fixture
def temp_input_file():
    filename = "example_data/debates/data.json"
    return filename


@pytest.fixture
def temp_output_file():
    return "logical_fallacy_opt.json"


@pytest.fixture
def temp_intermediate_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def map_config():
    return MapOp(
        name="identify_themes",
        type="map",
        prompt="""Analyze the following debate transcript and identify the main themes that are being discussed by the candidates. 
        """,
         output={"schema": {"fallacies": "list[{fallacy: string, description: string, quote: string}]"}},
        model="gpt-4o-mini",
    )

@pytest.fixture
def unnest_config():
    return UnnestOp(
        name = "unnest_fallacies",
        type = "unnest",
        unnest_key = "fallacies"
    )

@pytest.fixture
def resolve_config():
    return ResolveOp(
        name="fallacy_resolver",
        type="resolve",
        blocking_keys=["fallacies"],
        blocking_threshold=0.8,
        comparison_prompt="Compare the following two entries and determine if they likely refer to the same type of logical fallacy: Fallacy 1: {{ input1 }} Fallacy 2: {{ input2 }} Return true if they likely match, false otherwise.",
        output={"schema": {"fallacy": "string", "description": "string", "quote": "string"}},
        embedding_model="text-embedding-3-small",
        comparison_model="gpt-4o-mini",
        resolution_model="gpt-4o-mini",
        resolution_prompt="Given the following list of similar entries, determine the most comprehensive description of the logical fallacy. {{ inputs }}",
        batch_size = 100
    )

# def test_pipeline_optimization(
#     map_config, reduce_config, temp_input_file, temp_output_file, temp_intermediate_dir
# ):
#     pipeline = Pipeline(
#         name="test_pipeline",
#         datasets={"test_input": Dataset(type="file", path=temp_input_file)},
#         operations=[map_config],
#         steps=[
#             PipelineStep(
#                 name="map_step", input="test_input", operations=["find_logical_fallacies"]
#             )
#         ],
#         output=PipelineOutput(
#             type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
#         ),
#         default_model="gpt-4o-mini",
#     )

#     optimized_pipeline = pipeline.optimize(
#         max_threads=4, model="gpt-4o-mini", timeout=10
#     )

#     print(optimized_pipeline._to_dict())

#     print('starting run')

#     cost = optimized_pipeline.run(max_threads = 4)

def test_pipeline_map_unnest_resolve(
    map_config, unnest_config, resolve_config, temp_input_file, temp_output_file, temp_intermediate_dir
):
    pipeline = Pipeline(
        name="debate_analysis_pipeline",
        datasets={"debates": Dataset(type="file", path=temp_input_file)},
        operations=[map_config, unnest_config, resolve_config],
        steps=[
            PipelineStep(
                name="extract_fallacies",
                input="debates",
                operations=["find_logical_fallacies"]
            ),
            PipelineStep(
                name="unnest_fallacies",
                input="extract_fallacies",
                operations=["unnest_fallacies"]
            ),
            PipelineStep(
                name="resolve_similar_fallacies",
                input="unnest_fallacies", 
                operations=["fallacy_resolver"]
            )
        ],
        output=PipelineOutput(
            type="file",
            path="logical_fallacy_analysis.json",
            intermediate_dir="checkpoints"
        ),
        default_model="gpt-4o-mini",
        rate_limits={"llm_call": [{"count": 1, "per": 1, "unit": "second"}, {"count": 10, "per": 1, "unit": "minute"}]}  # Limit LLM calls to 1 per second or 10 per minute
    )

    cost = pipeline.run(max_threads=4)


