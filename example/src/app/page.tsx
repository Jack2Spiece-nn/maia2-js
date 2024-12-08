"use client";

import Maia from "./maia2/model";
import { useState } from "react";

export default function Home() {
  const model = new Maia({
    modelPath: "/maia_rapid_onnx.onnx",
  });
  const [fen, setFen] = useState("");
  const [output, setOutput] = useState<{ moveProbs: Record<string, number> }>();

  async function runModel() {
    try {
      const eloSelf = 1100;
      const eloOppo = 1100;

      const result = await model.evaluate(fen, eloSelf, eloOppo);

      setOutput(result);
    } catch (error) {
      console.error("Error running the model:", error);
    }
  }

  return (
    <div>
      <div>
        <h1>Chess Model Output</h1>
        <input
          type="text"
          value={fen}
          onChange={(e) => setFen(e.target.value)}
          placeholder="Enter FEN"
        />
        <button onClick={runModel}>Run Model</button>
        <pre>{output ? JSON.stringify(output, null, 2) : "Loading..."}</pre>
      </div>
    </div>
  );
}
