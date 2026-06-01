// swift-tools-version: 5.9
import PackageDescription
import Foundation

let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let swiftSDKRelativePath = "sdk/swift"
let ffiXCFrameworkRelativePath = "\(swiftSDKRelativePath)/Generated/ClosedMeshFFI.xcframework"
let ffiXCFrameworkPath = "\(repoRoot)/\(ffiXCFrameworkRelativePath)"
let remoteFFIXCFrameworkURL = "https://github.com/closedmesh/closedmesh-llm/releases/download/v0.66.61/ClosedMeshFFI.xcframework.zip"
let remoteFFIXCFrameworkChecksum = "a4f9ab20b6271d6caa0e6e0e1ad54488c1202690c4485adf6de7588949f6d831"
let forceStubFFI = ProcessInfo.processInfo.environment["MESH_SWIFT_FORCE_STUB"] == "1"
let hasLocalFFIXCFramework = FileManager.default.fileExists(atPath: ffiXCFrameworkPath)
let hasRemoteFFIXCFramework = !forceStubFFI
    && !remoteFFIXCFrameworkURL.contains("__MESH_SWIFT_RELEASE_TAG__")
    && !remoteFFIXCFrameworkChecksum.contains("__MESH_SWIFT_RELEASE_CHECKSUM__")

var meshLLMDependencies: [Target.Dependency] = []
var packageTargets: [Target] = []

if hasLocalFFIXCFramework {
    meshLLMDependencies.append("ClosedMeshFFI")
    packageTargets.append(
        .binaryTarget(
            name: "ClosedMeshFFI",
            path: ffiXCFrameworkRelativePath
        )
    )
} else if hasRemoteFFIXCFramework {
    meshLLMDependencies.append("ClosedMeshFFI")
    packageTargets.append(
        .binaryTarget(
            name: "ClosedMeshFFI",
            url: remoteFFIXCFrameworkURL,
            checksum: remoteFFIXCFrameworkChecksum
        )
    )
}

let hasFFIBinaryTarget = hasLocalFFIXCFramework || hasRemoteFFIXCFramework

let package = Package(
    name: "ClosedMesh",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(
            name: "ClosedMesh",
            targets: ["ClosedMesh"]
        ),
    ],
    targets: [
        .target(
            name: "ClosedMesh",
            dependencies: meshLLMDependencies,
            path: "sdk/swift/Sources/ClosedMesh",
            exclude: hasFFIBinaryTarget ? [] : ["Generated"],
            linkerSettings: [
                .linkedFramework("SystemConfiguration"),
            ]
        ),
        .testTarget(
            name: "ClosedMeshTests",
            dependencies: ["ClosedMesh"],
            path: "sdk/swift/Tests/ClosedMeshTests"
        ),
    ] + packageTargets
)
