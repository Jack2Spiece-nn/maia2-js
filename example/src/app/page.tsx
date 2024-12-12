"use client";

import { Chess } from "chess.js";
import Maia from "./maia2/model";
import { DrawShape } from "chessground/draw";
import { Dests, Key } from "chessground/types";
import "chessground/assets/chessground.base.css";
import "chessground/assets/chessground.brown.css";
import Chessground from "@react-chess/chessground";
import "chessground/assets/chessground.cburnett.css";
import { useState, useEffect, useCallback } from "react";
import Link from "next/link";

export default function Home() {
  const [loaded, setLoaded] = useState(false);
  const [selfElo, setSelfElo] = useState(1100);
  const [oppoElo, setOppoElo] = useState(1100);
  const [board, setBoard] = useState<Chess>(
    new Chess("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
  );
  const [arrows, setArrows] = useState<DrawShape[]>([]);
  const [model, setModel] = useState<Maia>();
  const [output, setOutput] = useState<{
    value: number;
    policy: Record<string, number>;
  }>();

  const dests: Dests = new Map();

  board.moves({ verbose: true }).forEach((move) => {
    const from = move.from as Key;
    const to = move.to as Key;
    if (!dests.has(from)) {
      dests.set(from, []);
    }
    dests.get(from)?.push(to);
  });

  useEffect(() => {
    const maia = new Maia({ modelPath: "/maia_rapid_onnx.onnx" });
    setModel(maia);
    maia.Ready.then((ready) => {
      if (ready) {
        setLoaded(true);
      }
    });
  }, []);

  const runModel = useCallback(async () => {
    try {
      if (!loaded || !model) return;
      setOutput(undefined);
      setArrows([]);

      const result = await model.evaluate(board.fen(), selfElo, oppoElo);

      result.policy = Object.fromEntries(
        Object.entries(result.policy).sort(([, a], [, b]) => b - a),
      );

      setOutput({ ...result });
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
  }, [board, model, oppoElo, selfElo, setOutput, loaded]);

  useEffect(() => {
    runModel();
  }, [runModel, selfElo, oppoElo]);

  return (
    <div className="flex w-screen flex-col items-center justify-start gap-4 bg-[#1C1A1E] py-6 md:h-screen md:justify-center md:gap-8 md:py-0">
      <div className="flex flex-col items-center gap-2">
        <div className="flex flex-col items-center justify-center gap-3 md:flex-row md:gap-6">
          <h1 className="order-2 text-center text-3xl font-bold text-white md:order-1 md:text-4xl">
            Maia2 ONNX Batch Example
          </h1>
          <div
            className={`order-1 rounded-md px-3 py-1 text-sm text-white md:order-2 ${loaded ? "bg-green-500" : "bg-red-500"}`}
          >
            <p>{loaded ? "READY" : "LOADING"}</p>
          </div>
        </div>
        <Link href="/batch">
          <p className="text-lg text-white underline">See Batch Performance</p>
        </Link>
      </div>
      <div className="flex w-full flex-col items-center justify-center gap-2 md:w-auto md:flex-row md:items-start">
        <div className="flex w-full flex-col items-center gap-2 md:w-auto">
          <div className="h-[90vw] w-[90vw] md:h-[50vh] md:w-[50vh]">
            <Chessground
              contained
              config={{
                fen: board.fen(),
                movable: {
                  free: false,
                  dests: loaded ? dests : new Map(),
                  events: {
                    after: (orig: Key, dest: Key) => {
                      const move = board.move({ from: orig, to: dest });
                      if (move) {
                        setBoard(board);

                        runModel();
                      }
                    },
                  },
                },
                drawable: {
                  autoShapes: arrows,
                },
              }}
            />
          </div>
          <div className="flex w-[90vw] flex-col gap-2 md:w-[50vh]">
            <div className="flex w-full items-center justify-between gap-2">
              <div className="flex w-full flex-1 flex-col">
                <label className="text-white">Self Elo</label>
                <select
                  value={selfElo}
                  onChange={(e) => setSelfElo(parseInt(e.target.value))}
                  className="h-10 rounded-sm bg-white/5 px-4 font-mono text-white/60 focus:outline-none"
                >
                  {[1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900].map(
                    (elo) => (
                      <option key={elo} value={elo}>
                        {elo}
                      </option>
                    ),
                  )}
                </select>
              </div>
              <div className="flex w-full flex-1 flex-col">
                <label className="text-white">Opponent Elo</label>
                <select
                  value={oppoElo}
                  onChange={(e) => setOppoElo(parseInt(e.target.value))}
                  className="h-10 rounded-sm bg-white/5 px-4 font-mono text-white/60 focus:outline-none"
                >
                  {[1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900].map(
                    (elo) => (
                      <option key={elo} value={elo}>
                        {elo}
                      </option>
                    ),
                  )}
                </select>
              </div>
            </div>
            <div className="h-10 w-full flex-1 flex-row items-center justify-center rounded-sm bg-white/5 px-4 py-2 font-mono text-white/60 focus:outline-none">
              <p className="whitespace-nowrap text-[0.5rem] md:text-xs">
                {board.fen()}
              </p>
            </div>
          </div>
        </div>
        <div className="flex w-[90vw] flex-col items-start overflow-hidden rounded border border-white/5 bg-[#26252D] md:w-[50vh]">
          {output ? (
            <div className="flex w-full flex-col text-white">
              <div className="flex w-full justify-between bg-red-400/80 p-4">
                <p className="font-bold text-white">Maia Win %</p>
                <p className="text-mono text-white">
                  {(output.value * 100).toFixed(1)}%
                </p>
              </div>
              <div className="flex w-full flex-col overflow-y-scroll md:max-h-[50vh]">
                {Object.entries(output.policy).map(([move, prob], index) => (
                  <div
                    key={index}
                    className={`flex w-full items-center justify-between px-4 py-1 ${
                      index % 2 === 0
                        ? "bg-[#9F4F44] bg-opacity-[0.02]"
                        : "bg-[#9F4F44] bg-opacity-10"
                    }`}
                  >
                    <p className="w-64 text-lg">
                      {new Chess(board.fen()).move(move)?.san}
                    </p>
                    <p className="font">{(prob * 100).toFixed(1)}%</p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex flex-col rounded-sm p-4 text-white">
              Waiting for analysis...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
