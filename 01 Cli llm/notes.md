# Assignment 01 — Raw LLM Engineering

> [!abstract] Goal
> Build a CLI LLM client **without an AI/agent framework** and understand what happens underneath LLM frameworks.

```text
Raw API → SDK → LangChain → LangGraph → Agents
```

---

## 1. LLM API Architecture

```text
CLI / Application
       ↓
   HTTP Request
       ↓
 Provider API
       ↓
 Model Inference
       ↓
 HTTP Response
       ↓
   Application
```

### Request contains

* **Endpoint** → API URL
* **Headers** → authentication + metadata
* **Payload** → model, messages, parameters
* **HTTP method** → usually `POST`

### Response contains

* HTTP status code
* Headers
* Response body
* Generated text
* Usage / metadata

---

## 2. HTTP / REST

### Common HTTP methods

| Method   | Purpose             |
| -------- | ------------------- |
| `GET`    | Retrieve            |
| `POST`   | Send/process/create |
| `PUT`    | Replace             |
| `PATCH`  | Partial update      |
| `DELETE` | Delete              |

LLM inference commonly uses:

```text
POST + JSON body
```

### Important status codes

```text
200 → Success
400 → Bad request
401 → Unauthorized
403 → Forbidden
404 → Not found
429 → Rate limited
500 → Server error
502/503/504 → Temporary server/gateway errors
```

---

## 3. Authentication

API keys are commonly sent through headers:

```text
Authorization: Bearer <API_KEY>
```

### Best practice

```text
Environment variable
        ↓
Application
        ↓
HTTP header
```

Never hard-code API keys.

```python
os.getenv("OPENAI_API_KEY")
```

---

## 4. JSON

LLM APIs generally communicate using JSON.

```text
Python dict
    ↓ serialize
   JSON
    ↓ HTTP
   API
    ↓
   JSON
    ↓ deserialize
Python dict
```

In `requests`:

```python
json=payload
```

→ Python dictionary is sent as JSON.

```python
response.json()
```

→ JSON response becomes a Python object.

---

## 5. Messages & Roles

Conversation is represented as a list of messages:

```json
[
  {
    "role": "system",
    "content": "You are helpful."
  },
  {
    "role": "user",
    "content": "Hello"
  },
  {
    "role": "assistant",
    "content": "Hi!"
  }
]
```

### Roles

| Role        | Purpose                 |
| ----------- | ----------------------- |
| `system`    | Instructions / behavior |
| `user`      | User input              |
| `assistant` | Previous model response |

Conversation history can therefore be maintained as a sequence of messages.

---

## 6. Tokens

LLMs process **tokens**, not simply words.

```text
Text
 ↓
Tokenizer
 ↓
Tokens
 ↓
Model
```

A token can represent:

* Part of a word
* A complete word
* Punctuation
* Whitespace/subword

### Remember

```text
More tokens
   ↓
More context usage
   ↓
Potentially more cost + latency
```

> Token count ≠ word count ≠ character count.

---

## 7. Context Window

The **context window** is the amount of context a model can handle for a request, subject to the model/provider limits.

```text
System instructions
        +
Conversation history
        +
User prompt
        +
Retrieved documents
        +
Tool results
        +
Output
        ↓
   Context Window
```

Context management becomes important for:

* Long conversations
* RAG
* Agents
* Memory
* Large documents

---

## 8. Generation Parameters

### Temperature

Controls randomness/variation.

```text
Lower temperature
→ more deterministic

Higher temperature
→ more variation
```

### Maximum output tokens

Limits the amount of generated output.

### Other parameters

Depending on provider/model:

* `top_p`
* stop sequences
* penalties
* seed
* reasoning-specific parameters

> Parameters and supported values can differ between providers/models.

---

# 9. Non-Streaming

Without streaming:

```text
Client
  ↓
Request
  ↓
Provider generates complete response
  ↓
Complete response
  ↓
Client
```

### Pros

* Simple implementation
* Easy response parsing

### Cons

* User waits for the response to finish
* Higher **perceived latency**

---

# 10. Streaming

With streaming:

```text
Client
  ↓
Request
  ↓
Provider
  ↓
Chunk 1
Chunk 2
Chunk 3
Chunk 4
  ↓
Client
```

Example:

```text
"Hel"
"lo "
"world"
"!"
```

Application combines the chunks:

```text
"Hello world!"
```

### Benefits

* Lower perceived latency
* Better CLI/chat UX
* Output can be processed while generation continues

---

## 11. SSE — Server-Sent Events

**SSE = Server-Sent Events**

A mechanism for sending events:

```text
Server → Client
```

over an HTTP connection.

Typical structure:

```text
event: <event-type>
data: <payload>
```

### Important distinction

```text
LLM Streaming
    ≠
SSE
```

**Streaming** = incremental response behavior.

**SSE** = one possible mechanism for delivering those incremental events.

---

## 12. Two `stream=True`s

### API-level

```json
"stream": true
```

Tells the **provider**:

> Return the generated response incrementally.

### `requests`-level

```python
requests.post(..., stream=True)
```

Tells the **HTTP client**:

> Allow the response body to be consumed incrementally instead of immediately reading the entire response.

### Mental model

```text
API stream=True
→ Provider streams

requests stream=True
→ Client reads incrementally
```

---

## 13. Streaming Chunks / Deltas

Streaming responses contain partial output.

```text
Request
  ↓
Event / Chunk
  ↓
Extract delta
  ↓
Display
  ↓
Next chunk
  ↓
...
  ↓
Complete
```

Different providers may use terms such as:

* `chunk`
* `delta`
* token
* event

> Don't assume every provider has the same streaming format.

---

## 14. `iter_lines()`

For raw `requests` streaming:

```python
response.iter_lines()
```

allows processing incoming data incrementally.

Conceptually:

```text
HTTP stream
     ↓
iter_lines()
     ↓
Incoming line/event
     ↓
Parse
     ↓
Extract generated text
     ↓
Display
```

---

## 15. TTFT

**TTFT = Time To First Token**

Time from starting the request until the first generated output is received.

```text
Request starts
      ↓
   waiting
      ↓
First output
      ↑
     TTFT
```

### Why it matters

Two important latency metrics:

```text
TTFT
→ How quickly does output start?

Total latency
→ How long until everything finishes?
```

A system can have:

```text
Low TTFT + High total latency
```

and still feel responsive to the user.

---

# 16. Timeouts

Without a timeout:

```text
Request
   ↓
Server hangs
   ↓
Client waits indefinitely
```

With timeout:

```text
Request
   ↓
Timeout reached
   ↓
Exception
   ↓
Handle / Retry / Fail
```

Example:

```python
timeout=30
```

### Purpose

Prevent indefinitely hanging requests.

---

# 17. Error Handling

Handle different failure categories separately.

### HTTP/API errors

```text
401 → Authentication problem
429 → Rate limit
5xx → Provider/server problem
```

### Network errors

```text
Timeout
ConnectionError
```

### Response errors

```text
Invalid JSON
Missing fields
Unexpected response structure
```

### General flow

```text
HTTP request
     ↓
Check HTTP success
     ↓
Parse response
     ↓
Validate expected fields
     ↓
Return result
```

---

## 18. `raise_for_status()`

```python
response.raise_for_status()
```

Checks HTTP status.

```text
2xx
 ↓
Continue

4xx / 5xx
 ↓
Raise HTTP exception
```

Therefore you usually don't need to manually check:

```text
if status_code >= 400
```

### Important

`raise_for_status()` checks **HTTP-level errors**.

It does **not** guarantee that the JSON response has the fields your application expects.

---

## 19. `e.response`

For an HTTP-related `requests` exception:

```python
e.response
```

may contain the original HTTP response.

Useful:

```python
e.response.status_code
e.response.text
e.response.headers
e.response.json()
```

### Important

For network failures where no HTTP response was received:

```text
e.response == None
```

So check before accessing it.

---

# 20. Retries

Temporary failures can sometimes be retried.

```text
Request
   ↓
Failure
   ↓
Retry
   ↓
Failure
   ↓
Retry
   ↓
Success
```

### Exponential backoff

Increase delay between attempts:

```text
1 sec
2 sec
4 sec
8 sec
...
```

Formula:

```text
delay = initial_delay × 2^attempt
```

### Usually retry

* Timeout
* Connection failure
* Rate limiting
* Temporary `5xx`

### Usually DON'T blindly retry

* Invalid request
* Authentication failure
* Permanent client errors

---

# 21. Logging

Logging provides operational visibility.

Useful information:

```text
Provider
Model
Request started
Latency
TTFT
Retry count
Error type
HTTP status
```

### Never log secrets

```text
❌ API keys
❌ Passwords
❌ Tokens
❌ Sensitive user data
```

---

# 22. Latency Measurement

Use a high-resolution timer such as:

```python
time.perf_counter()
```

### Measure

```text
Start
 ↓
Request
 ↓
First output → TTFT
 ↓
Complete output → Total latency
```

---

# 23. Provider Abstraction

### Without abstraction

```text
Application
    ↓
OpenAI-specific code
    ↓
Everything depends on OpenAI
```

Switching providers becomes difficult.

### With abstraction

```text
                 ┌→ OpenAI
Application → LLM Interface → Gemini
                 ├→ Groq
                 └→ Ollama
```

Common interface might expose:

```text
generate()
stream()
```

The application doesn't need to know provider-specific implementation details.

---

# 24. Provider-Specific Differences

Providers may differ in:

* API endpoint
* Authentication
* Model names
* Request schema
* Generation parameters
* Streaming format
* Response structure
* Error format
* Token accounting
* Context limits
* Available capabilities

Therefore:

> **Common interface ≠ identical API implementation**

The provider adapter handles these differences.

---

# 25. Recommended Architecture

```text
┌─────────────────────┐
│         CLI         │
│   Input / Output    │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│    LLM Interface    │
│ generate / stream   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Provider Adapter   │
│                     │
│ OpenAI / Ollama ... │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│     HTTP Client     │
│                     │
│ timeout / retry /   │
│ error handling      │
└──────────┬──────────┘
           ↓
       Provider API
```

### Responsibility separation

| Layer             | Responsibility                  |
| ----------------- | ------------------------------- |
| **CLI**           | User interaction                |
| **LLM Interface** | Common contract                 |
| **Provider**      | Provider-specific API + parsing |
| **HTTP Client**   | HTTP, timeout, retries, errors  |
| **Configuration** | Environment variables           |

---

# 26. End-to-End Mental Model

```text
User Input
    ↓
Create Message
    ↓
Build JSON Payload
    ↓
Add Authentication
    ↓
HTTP POST
    ↓
Provider API
    ↓
Model Inference
    ↓
┌──────────────────┐
│ Non-streaming    │
│ → Complete JSON  │
└────────┬─────────┘
         │
         OR
         │
┌────────▼─────────┐
│ Streaming        │
│ → Events/chunks  │
└────────┬─────────┘
         ↓
Parse Response
    ↓
Extract Text
    ↓
Measure Latency
    ↓
Display Output
```

---

# 27. Assignment 01 Checklist

* [ ] HTTP / REST
* [ ] POST request
* [ ] JSON request/response
* [ ] API authentication
* [ ] Environment variables
* [ ] Messages + roles
* [ ] Tokens
* [ ] Context windows
* [ ] Generation parameters
* [ ] Non-streaming
* [ ] Streaming
* [ ] SSE
* [ ] Chunks / deltas
* [ ] `iter_lines()`
* [ ] TTFT
* [ ] Total latency
* [ ] Timeouts
* [ ] Error handling
* [ ] `raise_for_status()`
* [ ] Retries
* [ ] Exponential backoff
* [ ] Logging
* [ ] Provider abstraction
* [ ] Multiple providers
* [ ] Provider-specific differences

---

# 28. Final Mental Model

> [!important] Core Idea
> **LLM frameworks are abstractions over lower-level API and engineering primitives.**

```text
HTTP
 ↓
LLM API
 ↓
SDK
 ↓
LangChain
 ↓
LangGraph
 ↓
Agents
 ↓
Production Agent System
```

### Assignment 01 teaches the bottom layer:

```text
HTTP + JSON
      +
Authentication
      +
Messages
      +
LLM parameters
      +
Streaming
      +
Retries / Timeouts
      +
Observability
      +
Provider abstraction
```

**Goal:** Don't memorize the API. Understand the pipeline.
