import XCTest
@testable import ClosedMesh

func makeOwnerKeypairBytesHex() -> String {
    #if canImport(ClosedMeshFFI)
    return generateOwnerKeypairHex()
    #else
    return "test-owner-keypair"
    #endif
}
