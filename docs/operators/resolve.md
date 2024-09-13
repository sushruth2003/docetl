# Resolve Operation

The Resolve operation in docetl is a powerful tool for identifying and merging duplicate entities in your data. It's particularly useful when dealing with inconsistencies that can arise from LLM-generated content or data from multiple sources.

## Motivation

Map operations executed by LLMs may sometimes yield inconsistent results, even when referring to the same entity. For example, when extracting patient names from medical transcripts, you might end up with variations like "Mrs. Smith" and "Jane Smith" for the same person. In such cases, a Resolve operation on the `patient_name` field can help standardize patient names before conducting further analysis.

## 🚀 Example: Standardizing Patient Names

Let's see a practical example of using the Resolve operation to standardize patient names extracted from medical transcripts.

```yaml
- name: standardize_patient_names
  type: resolve
  comparison_prompt: |
    Compare the following two patient name entries:

    Patient 1: {{ input1.patient_name }}
    Date of Birth 1: {{ input1.date_of_birth }}

    Patient 2: {{ input2.patient_name }}
    Date of Birth 2: {{ input2.date_of_birth }}

    Are these entries likely referring to the same patient? Consider name similarity and date of birth. Respond with "True" if they are likely the same patient, or "False" if they are likely different patients.
  resolution_prompt: |
    Standardize the following patient name entries into a single, consistent format:

    {% for entry in inputs %}
    Patient Name {{ loop.index }}: {{ entry.patient_name }}
    {% endfor %}

    Provide a single, standardized patient name that represents all the matched entries. Use the format "LastName, FirstName MiddleInitial" if available.
  output:
    schema:
      patient_name: string
```

This Resolve operation processes patient names to identify and standardize duplicates:

1. Compares all pairs of patient names using the `comparison_prompt`.
2. For identified duplicates, it applies the `resolution_prompt` to generate a standardized name.

Note: The prompt templates use Jinja2 syntax, allowing you to reference input fields directly (e.g., `input1.patient_name`).

!!! warning "Performance Consideration"

    You should not run this operation as-is unless your dataset is small! Running O(n^2) comparisons with an LLM can be extremely time-consuming for large datasets. Instead, optimize your pipeline first using `docetl build pipeline.yaml` and run the optimized version, which will generate efficient blocking rules for the operation.

## Blocking

To improve efficiency, the Resolve operation supports "blocking" - a technique to reduce the number of comparisons by only comparing entries that are likely to be matches. docetl supports two types of blocking:

1. Embedding similarity: Compare embeddings of specified fields and only process pairs above a certain similarity threshold.
2. Python conditions: Apply custom Python expressions to determine if a pair should be compared.

Here's an example of a Resolve operation with blocking:

```yaml
- name: standardize_patient_names
  type: resolve
  comparison_prompt: |
    # (Same as previous example)
  resolution_prompt: |
    # (Same as previous example)
  output:
    schema:
      patient_name: string
  blocking_keys:
    - last_name
    - date_of_birth
  blocking_threshold: 0.8
  blocking_conditions:
    - "len(left['last_name']) > 0 and len(right['last_name']) > 0"
    - "left['date_of_birth'] == right['date_of_birth']"
```

In this example, pairs will be considered for comparison if:

- The embedding similarity of their `last_name` and `date_of_birth` fields is above 0.8, OR
- Both entries have non-empty `last_name` fields AND their `date_of_birth` fields match exactly.

## Required Parameters

- `type`: Must be set to "resolve".
- `comparison_prompt`: The prompt template to use for comparing potential matches.
- `resolution_prompt`: The prompt template to use for reducing matched entries.
- `output`: Schema definition for the output from the LLM.

## Optional Parameters

| Parameter              | Description                                                                       | Default                       |
| ---------------------- | --------------------------------------------------------------------------------- | ----------------------------- |
| `embedding_model`      | The model to use for creating embeddings                                          | Falls back to `default_model` |
| `resolution_model`     | The language model to use for reducing matched entries                            | Falls back to `default_model` |
| `comparison_model`     | The language model to use for comparing potential matches                         | Falls back to `default_model` |
| `blocking_keys`        | List of keys to use for initial blocking                                          | All keys in the input data    |
| `blocking_threshold`   | Embedding similarity threshold for considering entries as potential matches       | None                          |
| `blocking_conditions`  | List of conditions for initial blocking                                           | []                            |
| `input`                | Specifies the schema or keys to subselect from each item to pass into the prompts | All keys from input items     |
| `embedding_batch_size` | The number of entries to send to the embedding model at a time                    | 1000                          |
| `compare_batch_size`   | The number of entity pairs processed in each batch during the comparison phase    | 100                           |
| `limit_comparisons`    | Maximum number of comparisons to perform                                          | None                          |

## Best Practices

1. **Anticipate Resolve Needs**: If you anticipate needing a Resolve operation and want to control the prompts, create it in your pipeline and let the optimizer find the appropriate blocking rules and thresholds.
2. **Let the Optimizer Help**: The optimizer can detect if you need a Resolve operation (e.g., because there's a downstream reduce operation you're optimizing) and can create a Resolve operation with suitable prompts and blocking rules.
3. **Effective Comparison Prompts**: Design comparison prompts that consider all relevant factors for determining matches.
4. **Detailed Resolution Prompts**: Create resolution prompts that effectively standardize or combine information from matched records.
5. **Appropriate Model Selection**: Choose suitable models for embedding (if used) and language tasks.

The Resolve operation is particularly useful for data cleaning, deduplication, and creating standardized records from multiple data sources. It can significantly improve data quality and consistency in your dataset.