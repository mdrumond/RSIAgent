// Keep fixture callers on the installed public entry point. No alternate
// dispatcher or resource discovery is installed by the test harness.
export async function portableCli(mainPath) {
  return mainPath;
}
