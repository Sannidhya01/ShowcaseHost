"use client";

import { useEffect, useState } from "react";
import { apiClient, ApiClientError } from "@/lib/api-client";
import type { HealthStatus } from "@/types/api";

type HealthState =
  | { state: "loading" }
  | { state: "healthy"; data: HealthStatus }
  | { state: "unavailable"; message: string };

export function ApiHealthStatus() {
  const [health, setHealth] = useState<HealthState>({ state: "loading" });

  useEffect(() => {
    let mounted = true;

    apiClient
      .getHealth()
      .then((data) => {
        if (mounted) {
          setHealth({ state: "healthy", data });
        }
      })
      .catch((error: unknown) => {
        if (!mounted) {
          return;
        }

        const message =
          error instanceof ApiClientError
            ? error.message
            : "Backend health endpoint is unreachable.";
        setHealth({ state: "unavailable", message });
      });

    return () => {
      mounted = false;
    };
  }, []);

  if (health.state === "loading") {
    return <p className="text-sm text-slate-600">Checking API health...</p>;
  }

  if (health.state === "unavailable") {
    return <p className="text-sm text-amber-700">{health.message}</p>;
  }

  return (
    <p className="text-sm text-emerald-700">
      API healthy: {health.data.service} {health.data.version}
    </p>
  );
}
