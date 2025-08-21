export function createPageUrl(name) {
  if (name === "Upload") return "/upload";
  if (name === "Dashboard") return "/dashboard";
  if (name === "Archive") return "/archive";
  return "/";
}
