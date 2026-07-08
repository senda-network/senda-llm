// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MeshExampleApp",
    platforms: [
        .macOS(.v13),
    ],
    dependencies: [
        .package(path: "../../../.."),
    ],
    targets: [
        .executableTarget(
            name: "MeshExampleApp",
            dependencies: [
                .product(name: "Senda", package: "senda-llm"),
            ],
            path: "Sources/MeshExampleApp"
        ),
    ]
)
