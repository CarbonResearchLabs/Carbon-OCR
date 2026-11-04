// carbon/http.h: a request's head parsed as carbon-ocr serve reads it, and the responses it writes.
#include "carbon/http.h"

#include <gtest/gtest.h>

#include <string>

namespace {

using carbon::http::BadRequest;
using carbon::http::parse_head;

int status_of(const std::string& head) {
  try {
    parse_head(head);
  } catch (const BadRequest& e) {
    return e.status;
  }
  return 0;
}

}  // namespace

TEST(Http, ReadsARequestsHead) {
  const std::string head =
      "POST /ocr?format=hocr&name=scan%201.png&x=a+b HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2080\r\n"
      "Content-Type: image/png\r\nExpect: 100-Continue\r\n\r\n";
  const auto r = parse_head(head + "the body");
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->first, head.size());  // the body starts after the blank line
  const carbon::http::Request& q = r->second;
  EXPECT_EQ(q.method, "POST");
  EXPECT_EQ(q.path, "/ocr");
  EXPECT_EQ(q.version, "HTTP/1.1");
  EXPECT_EQ(q.param("format").value_or(""), "hocr");
  EXPECT_EQ(q.param("name").value_or(""), "scan 1.png");
  EXPECT_EQ(q.param("x").value_or(""), "a b");
  EXPECT_FALSE(q.param("y").has_value());
  EXPECT_EQ(q.header("content-type").value_or(""), "image/png");
  EXPECT_TRUE(q.has_length);
  EXPECT_EQ(q.content_length, 2080u);
  EXPECT_TRUE(q.keep_alive);
  EXPECT_TRUE(q.expect_continue);
}

TEST(Http, WaitsForAWholeHead) {
  EXPECT_FALSE(parse_head("").has_value());
  EXPECT_FALSE(parse_head("GET /health HTTP/1.1\r\nHost: x\r\n").has_value());
  EXPECT_TRUE(parse_head("GET /health HTTP/1.1\r\n\r\n").has_value());
}

TEST(Http, KeepsAliveAsEachVersionSays) {
  EXPECT_TRUE(parse_head("GET / HTTP/1.1\r\n\r\n")->second.keep_alive);
  EXPECT_FALSE(parse_head("GET / HTTP/1.1\r\nConnection: close\r\n\r\n")->second.keep_alive);
  EXPECT_FALSE(parse_head("GET / HTTP/1.0\r\n\r\n")->second.keep_alive);
  EXPECT_TRUE(parse_head("GET / HTTP/1.0\r\nConnection: Keep-Alive\r\n\r\n")->second.keep_alive);
  EXPECT_FALSE(parse_head("GET / HTTP/1.1\r\n\r\n")->second.has_length);
}

TEST(Http, RefusesWhatItCannotRead) {
  EXPECT_EQ(status_of("GET\r\n\r\n"), 400);
  EXPECT_EQ(status_of("GET  / HTTP/1.1\r\n\r\n"), 400);
  EXPECT_EQ(status_of("GET health HTTP/1.1\r\n\r\n"), 400);  // a target is a path
  EXPECT_EQ(status_of("GET / HTTP/2.0\r\n\r\n"), 505);
  EXPECT_EQ(status_of("GET / FTP/1.0\r\n\r\n"), 400);
  EXPECT_EQ(status_of("GET / HTTP/1.1\r\nno colon\r\n\r\n"), 400);
  EXPECT_EQ(status_of("GET / HTTP/1.1\r\nBad Name: x\r\n\r\n"), 400);
  EXPECT_EQ(status_of("POST / HTTP/1.1\r\nContent-Length: 12x\r\n\r\n"), 400);
  EXPECT_EQ(status_of("POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\n"), 400);
  EXPECT_EQ(status_of("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"), 411);
  EXPECT_EQ(status_of("POST / HTTP/1.1\r\nExpect: something\r\n\r\n"), 417);
  EXPECT_EQ(status_of("GET /%zz HTTP/1.1\r\n\r\n"), 400);
  EXPECT_EQ(status_of("GET /?a=%4 HTTP/1.1\r\n\r\n"), 400);
  EXPECT_EQ(status_of("GET / HTTP/1.1\r\nX: " + std::string(20000, 'a')), 431);  // long, and no end yet
  EXPECT_EQ(status_of("GET / HTTP/1.1\r\nX: " + std::string(20000, 'a') + "\r\n\r\n"), 431);
  EXPECT_EQ(carbon::http::decode_url("a%2Fb+c", false), "a/b+c");
}

TEST(Http, WritesAResponse) {
  const std::string r = carbon::http::response(503, "text/plain", "busy\n", false, {{"Retry-After", "1"}});
  EXPECT_EQ(r,
            "HTTP/1.1 503 Service Unavailable\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n"
            "Connection: close\r\nRetry-After: 1\r\n\r\nbusy\n");
  EXPECT_EQ(carbon::http::reason(413), "Content Too Large");
}
