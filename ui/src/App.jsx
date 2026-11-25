import { useEffect, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

export default function App() {
  const [models, setModels] = useState([]);
  const [candidate, setCandidate] = useState("bpe500");

  const [prompt, setPrompt] = useState("<|song|>\n");

  const [maxNewTokens, setMaxNewTokens] = useState(150);
  const [temperature, setTemperature] = useState(0.85);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.95);

  const [repPenalty, setRepPenalty] = useState(1.15);
  const [repWindow, setRepWindow] = useState(64);
  const [slideKeep, setSlideKeep] = useState(64);

  const [stopAtEnd, setStopAtEnd] = useState(true);
  const [includeSpecials, setIncludeSpecials] = useState(false);
  const [stream, setStream] = useState(true);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [output, setOutput] = useState("");
  const [meta, setMeta] = useState(null);

  useEffect(() => {
    async function loadModels() {
      try {
        const res = await fetch(`${API_URL}/models`);
        const data = await res.json();
        setModels(data.models || []);

        if (data.default) {
          setCandidate(data.default);
        }
      } catch (e) {
        setError(`Cannot load models from ${API_URL}. Reason: ${e.message}`);
      }
    }

    loadModels();
  }, []);

  async function generate() {
    setLoading(true);
    setError(null);
    setOutput("");
    setMeta(null);

    const payload = {
      candidate,
      prompt,
      max_new_tokens: maxNewTokens,
      temperature,
      top_k: topK,
      top_p: topP,
      rep_penalty: repPenalty,
      rep_window: repWindow,
      slide_keep: slideKeep,
      stop_at_end: stopAtEnd,
      include_specials: includeSpecials,
      stream,
    };

    try {
      const res = await fetch(`${API_URL}/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail);
      }

      if (stream && res.headers.get("content-type")?.includes("text/event-stream")) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();

        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          const parts = buffer.split("\n\n");
          buffer = parts.pop() || "";

          for (const part of parts) {
            const line = part.trim();

            if (!line.startsWith("data:")) continue;

            const jsonText = line.slice(5).trim();
            if (!jsonText) continue;

            const event = JSON.parse(jsonText);

            if (event.type === "delta") {
              setOutput((prev) => prev + event.text);
            } else if (event.type === "replace") {
              setOutput(event.text);
            } else if (event.type === "done") {
              setMeta(event);
            }
          }
        }
      } else {
        const data = await res.json();
        setOutput(data.text || "");
        setMeta(data);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ fontFamily: "system-ui", maxWidth: 1000, margin: "0 auto", padding: 20 }}>
      <h1>Indio Tiny LM</h1>

      {error && (
        <pre style={{ color: "crimson", whiteSpace: "pre-wrap" }}>
          {error}
        </pre>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 20 }}>
        <div>
          <h3>Model</h3>

          <label>
            BPE candidate
            <select
              value={candidate}
              onChange={(e) => setCandidate(e.target.value)}
              style={{ width: "100%" }}
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.id} | vocab {m.vocab_size} | BPC {m.best_bpc.toFixed(3)}
                </option>
              ))}
            </select>
          </label>

          <h3>Prompt</h3>

          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={5}
            style={{ width: "100%" }}
          />

          <h3>Generation</h3>

          <label>
            max_new_tokens: {maxNewTokens}
            <input
              type="range"
              min={20}
              max={300}
              value={maxNewTokens}
              onChange={(e) => setMaxNewTokens(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>

          <label>
            temperature: {temperature.toFixed(2)}
            <input
              type="range"
              min={0.1}
              max={1.5}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>

          <label>
            top_k: {topK}
            <input
              type="range"
              min={1}
              max={100}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>

          <label>
            top_p: {topP.toFixed(2)}
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={topP}
              onChange={(e) => setTopP(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>

          <label>
            rep_penalty: {repPenalty.toFixed(2)}
            <input
              type="range"
              min={1}
              max={1.5}
              step={0.01}
              value={repPenalty}
              onChange={(e) => setRepPenalty(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>

          <label>
            rep_window: {repWindow}
            <input
              type="range"
              min={0}
              max={128}
              value={repWindow}
              onChange={(e) => setRepWindow(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>

          <label>
            slide_keep: {slideKeep}
            <input
              type="range"
              min={16}
              max={128}
              value={slideKeep}
              onChange={(e) => setSlideKeep(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>

          <label style={{ display: "block" }}>
            <input
              type="checkbox"
              checked={stopAtEnd}
              onChange={(e) => setStopAtEnd(e.target.checked)}
            />
            stop at end token
          </label>

          <label style={{ display: "block" }}>
            <input
              type="checkbox"
              checked={includeSpecials}
              onChange={(e) => setIncludeSpecials(e.target.checked)}
            />
            include special tokens
          </label>

          <label style={{ display: "block" }}>
            <input
              type="checkbox"
              checked={stream}
              onChange={(e) => setStream(e.target.checked)}
            />
            stream response
          </label>

          <button
            onClick={generate}
            disabled={loading}
            style={{ marginTop: 10, padding: "10px 16px" }}
          >
            {loading ? "Generating..." : "Generate"}
          </button>
        </div>

        <div>
          <h3>Output</h3>

          {meta && (
            <div style={{ fontSize: 12, opacity: 0.7 }}>
              {JSON.stringify(meta)}
            </div>
          )}

          <pre
            style={{
              whiteSpace: "pre-wrap",
              border: "1px solid #ddd",
              borderRadius: 8,
              padding: 12,
              minHeight: 300,
            }}
          >
            {output}
          </pre>
        </div>
      </div>
    </div>
  );
}