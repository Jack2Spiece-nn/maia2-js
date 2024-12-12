"use client";

import Maia from "../maia2/model";
import openings from "./openings.json";
import { Key } from "chessground/types";
import { DrawShape } from "chessground/draw";
import "chessground/assets/chessground.base.css";
import "chessground/assets/chessground.brown.css";
import Chessground from "@react-chess/chessground";
import "chessground/assets/chessground.cburnett.css";
import { useState, useEffect, useCallback } from "react";

const BOARDS = openings.slice(0, 100);

export default function Home() {
  const [model, setModel] = useState<Maia>();
  const [count, setCount] = useState(100);
  const [loaded, setLoaded] = useState(false);
  const [totalTime, setTotalTime] = useState(0);
  const [inferenceTime, setInferenceTime] = useState(0);

  const [boards, setBoards] = useState<
    {
      fen: string;
      selfElo: number;
      oppoElo: number;
      arrows: DrawShape[];
    }[]
  >(
    openings.slice(0, count).map((fen) => ({
      fen,
      selfElo: 1100,
      oppoElo: 1100,
      arrows: [],
    })),
  );

  useEffect(() => {
    const maia = new Maia({ modelPath: "/maia_rapid_onnx.onnx" });
    setModel(maia);
    maia.Ready.then((ready) => {
      if (ready) {
        setLoaded(true);
      }
    });
  }, []);

  useEffect(() => {
    setBoards(
      openings.slice(0, count).map((fen) => ({
        fen,
        selfElo: 1100,
        oppoElo: 1100,
        arrows: [],
      })),
    );
  }, [count]);

  const run = useCallback(async () => {
    const start = performance.now();
    const evaluation = await model?.batchEvaluate(
      boards.map((b) => b.fen),
      Array(BOARDS.length).fill(1100),
      Array(BOARDS.length).fill(1100),
    );

    if (!evaluation) {
      return;
    }

    const { result, time } = evaluation;

    setBoards((prev) => {
      return prev.map((board, i) => {
        if (!result?.[i].policy) {
          return board;
        }

        let maxKey;
        try {
          maxKey = Object.keys(result[i].policy).reduce((a, b) =>
            result[i].policy[a] > result[i].policy[b] ? a : b,
          );
        } catch {
          return board;
        }

        return {
          ...board,
          arrows: result?.[i].policy
            ? [
                {
                  brush: "red",
                  orig: maxKey.slice(0, 2) as Key,
                  dest: maxKey.slice(2) as Key,
                },
              ]
            : [],
        };
      });
    });

    const end = performance.now();
    setTotalTime(end - start);
    setInferenceTime(time);
  }, [model, boards]);

  return (
    <div className="flex w-screen flex-col items-center justify-start gap-4 bg-[#1C1A1E] py-6 md:justify-start md:gap-8 md:py-20">
      <div className="flex flex-col items-center justify-center gap-4">
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
        <div className="flex items-center gap-5">
          <p className="text-white">Boards:</p>
          <input
            className="rounded-sm bg-white/50 px-2 py-1 focus:outline-none"
            value={(Number.isNaN(count) ? 1 : count).toString()}
            onChange={(e) => {
              const newCount = Math.min(
                Math.max(parseInt(e.target.value ?? "1") ?? 1, 1),
                1000,
              );

              setCount(Number.isNaN(newCount) ? 1 : newCount);
            }}
          />
        </div>
        <button
          onClick={run}
          disabled={!loaded}
          className="w-full rounded bg-orange-400 px-10 py-2 hover:bg-orange-500"
        >
          <p className="text-xl font-bold text-white">Run</p>
        </button>
        <div className="flex flex-row items-center gap-3">
          <p className="text-white">
            Inference Time: <code>{inferenceTime.toFixed(2)}ms</code>
          </p>
          <span className="text-white">x</span>
          <p className="text-white">
            Total Time: <code>{totalTime.toFixed(2)}ms</code>
          </p>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-5 md:grid-cols-5">
        {boards.map((board, i) => (
          <div key={i} className="h-48 w-48 md:h-40 md:w-40">
            <Chessground
              contained={true}
              config={{
                fen: board.fen,
                coordinates: false,
                drawable: {
                  autoShapes: board.arrows,
                },
              }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
