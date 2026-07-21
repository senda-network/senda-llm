// swift-tools-version: 5.9
import PackageDescription
import Foundation

let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let swiftSDKRelativePath = "sdk/swift"
let ffiXCFrameworkRelativePath = "\(swiftSDKRelativePath)/Generated/SendaFFI.xcframework"
let ffiXCFrameworkPath = "\(repoRoot)/\(ffiXCFrameworkRelativePath)"
let remoteFFIXCFrameworkURL = "https://github.com/senda-network/senda-llm/releases/download/v0.66.97/SendaFFI.xcframework.zip"
let remoteFFIXCFrameworkChecksum = "79b7f6d4437118a7667412f780241c3466de30fb635914fb3e2a91a155044ecb"
let forceStubFFI = ProcessInfo.processInfo.environment["MESH_SWIFT_FORCE_STUB"] == "1"
let hasLocalFFIXCFramework = FileManager.default.fileExists(atPath: ffiXCFrameworkPath)
let hasRemoteFFIXCFramework = !forceStubFFI
    && !remoteFFIXCFrameworkURL.contains("__MESH_SWIFT_RELEASE_TAG__")
    && !remoteFFIXCFrameworkChecksum.contains("__MESH_SWIFT_RELEASE_CHECKSUM__")

var meshLLMDependencies: [Target.Dependency] = []
var packageTargets: [Target] = []

if hasLocalFFIXCFramework {
    meshLLMDependencies.append("SendaFFI")
    packageTargets.append(
        .binaryTarget(
            name: "SendaFFI",
            path: ffiXCFrameworkRelativePath
        )
    )
} else if hasRemoteFFIXCFramework {
    meshLLMDependencies.append("SendaFFI")
    packageTargets.append(
        .binaryTarget(
            name: "SendaFFI",
            url: remoteFFIXCFrameworkURL,
            checksum: remoteFFIXCFrameworkChecksum
        )
    )
}

let hasFFIBinaryTarget = hasLocalFFIXCFramework || hasRemoteFFIXCFramework

let package = Package(
    name: "Senda",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(
            name: "Senda",
            targets: ["Senda"]
        ),
    ],
    targets: [
        .target(
            name: "Senda",
            dependencies: meshLLMDependencies,
            path: "sdk/swift/Sources/Senda",
            exclude: hasFFIBinaryTarget ? [] : ["Generated"],
            linkerSettings: [
                .linkedFramework("SystemConfiguration"),
            ]
        ),
        .testTarget(
            name: "SendaTests",
            dependencies: ["Senda"],
            path: "sdk/swift/Tests/SendaTests"
        ),
    ] + packageTargets
)
