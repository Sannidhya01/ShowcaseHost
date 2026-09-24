import { writeFile } from "node:fs/promises";

const content = `// Placeholder for generated OpenAPI TypeScript types.
// Run this script after the API is running when contract generation is wired in.
export {};
`;

await writeFile(new URL("../packages/contracts/src/generated.ts", import.meta.url), content);
console.log("Wrote packages/contracts/src/generated.ts");
