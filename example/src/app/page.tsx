"use client";

import { Chess } from "chess.js";
import Maia from "./maia2/model";
import { useState, useEffect } from "react";
import Chessground from "@react-chess/chessground";
import "chessground/assets/chessground.base.css";
import "chessground/assets/chessground.brown.css";
import "chessground/assets/chessground.cburnett.css";
import { DrawShape } from "chessground/draw";
import { Key } from "chessground/types";

export default function Home() {
  const [selfElo, setSelfElo] = useState(1100);
  const [oppoElo, setOppoElo] = useState(1100);
  const [arrows, setArrows] = useState<DrawShape[]>([]);
  const [fen, setFen] = useState(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
  );
  const [model, setModel] = useState<Maia>();
  const [output, setOutput] = useState<{
    fen: string;
    value: number;
    policy: Record<string, number>;
  }>();

  useEffect(() => {
    setModel(new Maia({ modelPath: "/maia_rapid_onnx.onnx" }));
  }, []);

  async function runModel() {
    try {
      if (!model) return;

      const result = await model.evaluate(fen, selfElo, oppoElo);

      result.policy = Object.fromEntries(
        Object.entries(result.policy).sort(([, a], [, b]) => b - a)
      );

      setOutput({ ...result, fen });
      const top = Object.keys(result.policy)[0];

      setArrows([
        {
          brush: "red",
          orig: top.slice(0, 2) as Key,
          dest: top.slice(2) as Key,
        },
      ]);
    } catch (error) {
      console.error("Error running the model:", error);
    }
  }

  return (
    <div className="flex w-screen flex-col gap-4 md:gap-8 py-6 md:py-0 justify-start items-center md:justify-center md:h-screen bg-[#1C1A1E]">
      <h1 className="text-4xl font-bold text-white">Maia2 ONNX Example</h1>
      <div className="flex flex-col md:flex-row gap-2 items-start justify-center">
        <div className="flex flex-col gap-2">
          <div className="md:h-[50vh] md:w-[50vh] w-[90vw] h-[90vw]">
            <Chessground
              contained
              config={{
                fen: output?.fen,
                drawable: {
                  autoShapes: arrows,
                },
              }}
            />
          </div>
          <div className="flex flex-col gap-2 w-full">
            <div className="flex  gap-2 items-center w-full justify-between">
              <div className="flex flex-col w-full flex-1">
                <label className="text-white">Self Elo</label>
                <select
                  value={selfElo}
                  onChange={(e) => setSelfElo(parseInt(e.target.value))}
                  className="h-10 bg-white/5 text-white/60 px-4 font-mono focus:outline-none rounded-sm"
                >
                  {[1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900].map(
                    (elo) => (
                      <option key={elo} value={elo}>
                        {elo}
                      </option>
                    )
                  )}
                </select>
              </div>
              <div className="flex flex-col w-full flex-1">
                <label className="text-white">Opponent Elo</label>
                <select
                  value={oppoElo}
                  onChange={(e) => setOppoElo(parseInt(e.target.value))}
                  className="h-10 bg-white/5 text-white/60 px-4 font-mono focus:outline-none rounded-sm"
                >
                  {[1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900].map(
                    (elo) => (
                      <option key={elo} value={elo}>
                        {elo}
                      </option>
                    )
                  )}
                </select>
              </div>
            </div>
            <div className="flex items-center justify-center rounded-sm overflow-hidden">
              <input
                type="text"
                value={fen}
                placeholder="Enter FEN"
                className="flex-1 flex-row w-full h-10 bg-white/5 text-white/60 px-4 font-mono focus:outline-none"
                onChange={(e) => setFen(e.target.value)}
              />
              <button
                onClick={runModel}
                disabled={!model}
                className="p-2 bg-red-400 focus:outline-none hover:bg-red-500 cursor-pointer text-white px-4"
              >
                Analyze
              </button>
            </div>
          </div>
        </div>
        <div className="flex flex-col items-start w-full md:w-[50vh] bg-[#26252D] border border-white/5 rounded overflow-hidden">
          {output ? (
            <div className="flex flex-col w-full text-white">
              <div className="justify-between flex w-full p-4 bg-red-400/80">
                <p className="font-bold text-white">Maia Win %</p>
                <p className="text-mono text-white">
                  {(output.value * 100).toFixed(1)}%
                </p>
              </div>
              <div className="flex flex-col w-full md:max-h-[50vh] overflow-y-scroll">
                {Object.entries(output.policy).map(([move, prob], index) => (
                  <div
                    key={index}
                    className={`flex items-center justify-between w-full px-4 py-1 ${
                      index % 2 === 0
                        ? "bg-[#9F4F44] bg-opacity-[0.02]"
                        : "bg-[#9F4F44] bg-opacity-10"
                    }`}
                  >
                    <p className="w-64 text-lg">
                      {new Chess(output.fen).move(move)?.san}
                    </p>
                    <p className="font">{(prob * 100).toFixed(1)}%</p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="p-4 flex flex-col rounded-sm">
              Waiting for analysis...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
