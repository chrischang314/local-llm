import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const assetRoot = path.join(rootDir, "frontend", "vendor");
const manifestPath = path.join(rootDir, "scripts", "frontend-vendor-assets.json");
const packageJsonPath = path.join(rootDir, "package.json");
const packageLockPath = path.join(rootDir, "package-lock.json");

const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, "utf8"));
const packageLock = JSON.parse(fs.readFileSync(packageLockPath, "utf8"));

function resolveProjectPath(relativePath) {
  return path.join(rootDir, ...relativePath.split("/"));
}

function assertInside(childPath, parentPath) {
  const relative = path.relative(parentPath, childPath);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    const parentName = path.relative(rootDir, parentPath);
    const childName = path.relative(rootDir, childPath);
    throw new Error(`Refusing to write outside ${parentName}: ${childName}`);
  }
}

function assertPinnedVersion(asset) {
  const declaredVersion = packageJson.devDependencies?.[asset.packageName];
  if (declaredVersion !== asset.version) {
    throw new Error(
      `${asset.name} manifest version ${asset.version} does not match package.json devDependency ${declaredVersion}`,
    );
  }

  const lockEntry = packageLock.packages?.[`node_modules/${asset.packageName}`];
  if (lockEntry?.version !== asset.version) {
    throw new Error(
      `${asset.name} manifest version ${asset.version} does not match package-lock.json entry ${lockEntry?.version}`,
    );
  }
}

for (const asset of manifest.assets) {
  assertPinnedVersion(asset);

  const source = resolveProjectPath(asset.source);
  const destination = resolveProjectPath(asset.destination);
  assertInside(destination, assetRoot);

  if (!fs.existsSync(source)) {
    throw new Error(`Missing source asset: ${path.relative(rootDir, source)}`);
  }

  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.copyFileSync(source, destination);
}

console.log(`Copied ${manifest.assets.length} frontend assets to ${path.relative(rootDir, assetRoot)}`);
