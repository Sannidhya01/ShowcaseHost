import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiHealthStatus } from "./api-health-status";

describe("ApiHealthStatus", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders backend health when the API responds", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ status: "ok", service: "showcasehost-api", version: "0.1.0" }),
      })),
    );

    render(<ApiHealthStatus />);

    expect(screen.getByText("Checking API health...")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText("API healthy: showcasehost-api 0.1.0")).toBeInTheDocument();
    });
  });
});
