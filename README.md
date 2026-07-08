# LangGraph — Complete Beginner to Advanced Guide

> A step-by-step knowledge transfer on **LangGraph** — from core concepts to production-grade agentic patterns, with runnable code examples.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20AI-green)
![Level](https://img.shields.io/badge/Level-Beginner%20to%20Advanced-orange)

---

## Table of Contents

1. [What is LangGraph?](#-1-what-is-langgraph)
2. [State — The Heart of LangGraph](#-2-state--the-heart-of-langgraph)
3. [TypedDict — Defining State Shape](#-3-typeddict--defining-state-shape)
4. [Annotated & Reducers](#-4-annotated--reducers)
5. [Types of Reducers](#-5-types-of-reducers)
6. [Node](#-6-node)
7. [Building a StateGraph](#-7-building-a-stategraph)
8. [Edge vs Conditional Edge](#-8-edge-vs-conditional-edge)
9. [Conditional Edge — Line by Line](#-9-conditional-edge--line-by-line)
10. [Loops (Cyclic Graphs)](#-10-loops-cyclic-graphs)
11. [Checkpointing / Memory](#-11-checkpointing--memory)
12. [Human-in-the-Loop](#-12-human-in-the-loop)
13. [Parallel Execution (Fan-out / Fan-in)](#-13-parallel-execution-fan-out--fan-in)
14. [Subgraphs](#-14-subgraphs)
15. [Tool-Calling Agent (ReAct Pattern)](#-15-tool-calling-agent-react-pattern)
16. [Streaming](#-16-streaming)
17. [Cheat Sheet / Summary](#-17-cheat-sheet--summary)

---

## 1. What is LangGraph?

If you've used **LangChain**, you know it's great for linear pipelines: `A → B → C`.

But real agentic workflows need more — **branching**, **looping**, **retrying**, and **shared memory** across steps. That's exactly what **LangGraph** solves.

> Think of it like a flowchart you'd draw on paper — boxes (**nodes**) doing work, arrows (**edges**) deciding where to go next, and a **shared state** flowing through the whole thing.

### Install

```bash
uv add langgraph langchain-core
```

---

## 2. State — The Heart of LangGraph

**State** is a shared object that flows through every node in the graph. Every node can **read** it and **update** it.

```python
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]
    question: str
    answer: str
```

---

## 3. TypedDict — Defining State Shape

`TypedDict` defines a **fixed schema** for a dictionary — like a lightweight class saying "this dict will always have these keys with these types."

```python
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    answer: str
    count: int
```

### Why TypedDict over `dict` or `Pydantic`?

| Option | Pros / Cons |
|---|---|
| Plain `dict` | No type safety — typos in keys go unnoticed |
| `TypedDict` | IDE autocomplete + type checks, **zero runtime overhead** |
| `Pydantic BaseModel` | Runtime validation, but adds performance overhead per node call |

Since state passes through nodes constantly, `TypedDict` is preferred for speed. At runtime, it's literally just a normal Python `dict`.

---

## 4. Annotated & Reducers

`Annotated` attaches **extra metadata** to a type without changing the type itself. In LangGraph, it's used to define a **reducer** — the function that controls *how* a node's return value merges into existing state.

```python
from typing import Annotated
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]   # merge behavior = append
    answer: str                       # no Annotated = overwrite (default)
```

**Without a reducer (default):**
```python
def n1(state): return {"answer": "first"}
def n2(state): return {"answer": "second"}
# final state["answer"] = "second"  → n1's result is lost
```

**With `Annotated[list, add]`:**
```python
def n1(state): return {"messages": ["hi"]}
def n2(state): return {"messages": ["how are you"]}
# final state["messages"] = ["hi", "how are you"]  → both kept
```

> **Rule of thumb:** `TypedDict` = shape of state. `Annotated` = merge behavior per key.

---

## 5. Types of Reducers

| Reducer | Behavior | Typical Use Case |
|---|---|---|
| **None** (default) | Overwrite old value | `status`, `current_step`, `final_answer` |
| **`operator.add`** | Concatenate / append | logs, simple accumulating lists |
| **`add_messages`** | Append + update-by-message-id | chat history / agent messages |
| **Custom function** | Any logic you define | max score, dict merge, dedupe |
| **`Send`** (+ `add`) | Dynamic parallel fan-out & collect | map-reduce over unknown-size lists |

### Custom Reducer Example

```python
def keep_max(existing: int, new: int) -> int:
    return max(existing, new)

class State(TypedDict):
    best_score: Annotated[int, keep_max]   # always keeps the highest value seen
```

### Prebuilt Chat Reducer

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]   # smart: updates by ID, else appends
```

### Dynamic Fan-out with `Send`

```python
from langgraph.types import Send

class OverallState(TypedDict):
    topics: list[str]
    results: Annotated[list, add]

def continue_to_analysis(state: OverallState):
    return [Send("analyze_topic", {"topic": t}) for t in state["topics"]]

builder.add_conditional_edges("start", continue_to_analysis, ["analyze_topic"])
```

---

## 6. Node

A **node** is just a Python function. It takes `state` as input and returns a **partial dict** — only the keys it wants to update.

```python
def get_question(state: AgentState) -> AgentState:
    return {"question": "What is LangGraph?"}

def answer_question(state: AgentState) -> AgentState:
    ans = f"Answer to: {state['question']}"
    return {"answer": ans}
```

> A node should **never** return the entire state — only the fields it's changing.

---

## 7. Building a StateGraph

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)

builder.add_node("get_q", get_question)
builder.add_node("answer_q", answer_question)

builder.add_edge(START, "get_q")
builder.add_edge("get_q", "answer_q")
builder.add_edge("answer_q", END)

graph = builder.compile()

result = graph.invoke({"messages": [], "question": "", "answer": ""})
print(result)
```

- `START` and `END` are special sentinel nodes marking entry and exit points.
- `.compile()` converts your builder into a runnable graph.

---

## 8. Edge vs Conditional Edge

| Type | Behavior |
|---|---|
| **Edge** | Fixed transition — A always goes to B |
| **Conditional Edge** | Branching — next node decided at runtime based on state |

```python
# Normal edge
builder.add_edge("get_q", "answer_q")

# Conditional edge
def route_decision(state: AgentState) -> str:
    if "quantum" in state["question"].lower():
        return "quantum_node"
    return "general_node"

builder.add_conditional_edges(
    "get_q",
    route_decision,
    {
        "quantum_node": "quantum_node",
        "general_node": "general_node",
    }
)
```

---

## 9. Conditional Edge — Line by Line

```python
def route_decision(state: AgentState) -> str:
    if "quantum" in state["question"].lower():
        return "quantum_node"
    else:
        return "general_node"
```

- A **normal function**, not a node — it doesn't update state, it only **decides where to go next**.
- Must return a **string** — acting as a routing "label".
- `.lower()` makes the match case-insensitive.

```python
builder.add_conditional_edges(
    "get_q",           # 1️⃣ source node — after this finishes, route conditionally
    route_decision,     # 2️⃣ the routing function itself
    {
        "quantum_node": "quantum_node",   # 3️⃣ label → actual node name mapping
        "general_node": "general_node",
    }
)
```

### Runtime Flow

1. `get_q` node runs → updates state.
2. LangGraph calls `route_decision(state)`.
3. Function returns a label string (e.g. `"quantum_node"`).
4. LangGraph looks up that label in the mapping dict → finds the real node name.
5. Execution jumps to that node.

> **Analogy:** Like an SCCM Task Sequence condition step — "if OS is Windows 11 → Task Group A, else → Task Group B."

---

## 10. Loops (Cyclic Graphs)

This is where LangGraph shines over plain LangChain — you can loop back to a previous node.

```python
def check_quality(state: AgentState) -> str:
    if len(state["answer"]) < 20:
        return "retry"
    return "done"

builder.add_conditional_edges(
    "answer_q",
    check_quality,
    {
        "retry": "answer_q",   # loops back to itself
        "done": END
    }
)
```

> Always define an exit condition — otherwise it's an infinite loop. LangGraph raises `GraphRecursionError` after the default limit (25 steps), configurable via `recursion_limit`.

---

## 11. Checkpointing / Memory

Checkpointers save a state snapshot after every node execution — enabling **persistent memory** across turns/sessions.

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-123"}}

graph.invoke({"messages": [{"role": "user", "content": "Hi"}]}, config=config)
graph.invoke({"messages": [{"role": "user", "content": "What did I say?"}]}, config=config)
```

- `thread_id` = unique session ID; each thread keeps its own state history.
- For production, use `SqliteSaver` / `PostgresSaver` instead of `MemorySaver` (in-memory only).

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
```

---

## 12. Human-in-the-Loop

Pause execution before/after a critical node — e.g. requiring approval before an agent sends an email or deletes data.

```python
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["answer_q"]   # pause BEFORE this node runs
)

graph.invoke({"question": "Delete all files"}, config=config)  # stops before answer_q

# Inspect / modify state, then resume
graph.update_state(config, {"question": "Modified question"})
graph.invoke(None, config=config)   # None = resume from where it paused
```

`interrupt_after` is also available — pauses right after a node finishes.

---

## 13. Parallel Execution (Fan-out / Fan-in)

Multiple nodes can run **concurrently** from one source, then merge results.

```python
class ParallelState(TypedDict):
    topic: str
    result_a: str
    result_b: str
    summary: str

def node_a(state): return {"result_a": f"analysis A on {state['topic']}"}
def node_b(state): return {"result_b": f"analysis B on {state['topic']}"}
def merge(state): return {"summary": state["result_a"] + " + " + state["result_b"]}

builder = StateGraph(ParallelState)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("merge", merge)

builder.add_edge(START, "a")
builder.add_edge(START, "b")     # fan-out
builder.add_edge("a", "merge")
builder.add_edge("b", "merge")   # fan-in — waits for both
builder.add_edge("merge", END)
```

---

## 14. Subgraphs

A compiled graph can be nested **inside** another graph as a single node — great for modular, reusable workflows.

```python
# Build & compile a smaller graph
sub_builder = StateGraph(AgentState)
sub_builder.add_node("step1", get_question)
sub_builder.add_edge(START, "step1")
sub_builder.add_edge("step1", END)
sub_graph = sub_builder.compile()

# Use it as a node in the parent graph
parent_builder = StateGraph(AgentState)
parent_builder.add_node("sub_workflow", sub_graph)   # compiled graph as a node!
parent_builder.add_edge(START, "sub_workflow")
parent_builder.add_edge("sub_workflow", END)
```

---

## 15. Tool-Calling Agent (ReAct Pattern)

The most common real-world LangGraph pattern: the LLM decides to call a tool, the tool executes, results loop back to the LLM — until a final answer is ready.

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_aws import ChatBedrock

@tool
def get_weather(city: str) -> str:
    """Get weather for a city"""
    return f"{city} is sunny, 32°C"

tools = [get_weather]
llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-...").bind_tools(tools)

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: MessagesState) -> str:
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END

builder = StateGraph(MessagesState)   # prebuilt state with 'messages' + reducer
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))   # prebuilt node, auto-executes tool calls

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")   # loop back after tool execution

graph = builder.compile()
```

**Flow:** `agent` → decides tool needed → `tools` executes → back to `agent` → repeats until the LLM responds with no tool call → `END`.

> This is exactly what LangGraph's prebuilt `create_react_agent` does internally.

---

## 16. Streaming

```python
for chunk in graph.stream(
    {"messages": [("user", "What's weather in Chennai?")]},
    stream_mode="values"
):
    print(chunk)
```

| `stream_mode` | Returns |
|---|---|
| `"values"` | Full state after each step |
| `"updates"` | Only the diffs/changes |
| `"messages"` | Token-by-token streaming (chat apps) |

---

## 17. Cheat Sheet / Summary

| Concept | Purpose |
|---|---|
| **State** | Shared data flowing through the graph |
| **TypedDict** | Defines the shape/schema of state |
| **Annotated** | Attaches a reducer (merge behavior) to a state key |
| **Reducer** | Controls how updates merge into existing state |
| **Node** | Function that reads/updates state |
| **Edge** | Fixed transition between nodes |
| **Conditional Edge** | Runtime branching logic |
| **Loop** | Cycle back to an earlier node |
| **Checkpointer** | Persists state across calls (memory) |
| **interrupt_before / after** | Human-in-the-loop pause points |
| **Fan-out / Fan-in** | Parallel node execution |
| **Subgraph** | Nested, reusable compiled graph |
| **ToolNode** | Prebuilt node that executes tool calls |
| **MessagesState** | Prebuilt state schema for chat/agent apps |
| **Send** | Dynamic parallel fan-out (map-reduce style) |

---

## Quick Recap Flow

```
START → Node → Conditional Edge → (Loop back?) → Node → END
              ↑___________________________________|
```

---

### Author Notes

Compiled as part of hands-on LangGraph learning while building agentic AI projects (RAG pipelines, EDA automation agents, and tool-calling assistants).

Feel free to this repo if it helped you get started with LangGraph!
