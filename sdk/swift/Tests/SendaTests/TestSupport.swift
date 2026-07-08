import XCTest
@testable import Senda

func makeOwnerKeypairBytesHex() -> String {
    #if canImport(SendaFFI)
    return generateOwnerKeypairHex()
    #else
    return "test-owner-keypair"
    #endif
}
