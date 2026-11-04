// HTTP/1.1 for carbon-ocr serve (carbon/server.h): a request's head parsed, a response written. A body is
// read by its Content-Length alone (a chunked one is refused, 411), and a connection's requests one after
// another. Nothing here touches a socket.
#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace carbon::http {

struct Request {
  std::string method, target, version;  // the request line's three parts
  std::string path;                      // the target before its '?', decoded
  std::vector<std::pair<std::string, std::string>> query;    // the target's after it, decoded ('+' a space)
  std::vector<std::pair<std::string, std::string>> headers;  // in order, their names in lower case
  bool has_length = false;  // a Content-Length was given
  size_t content_length = 0;
  bool keep_alive = true;        // HTTP/1.1 unless Connection: close; HTTP/1.0 with Connection: keep-alive
  bool expect_continue = false;  // Expect: 100-continue: the client waits for 100 before its body

  // A header's value by its name in lower case (the first, if given twice), or nullopt.
  std::optional<std::string_view> header(std::string_view lower_name) const;
  // A query parameter's value (the first), or nullopt.
  std::optional<std::string_view> param(std::string_view name) const;
};

// A request that cannot be read: the status to answer with, and why.
class BadRequest : public std::runtime_error {
 public:
  BadRequest(int status, const std::string& why) : std::runtime_error(why), status(status) {}
  int status;
};

// The request whose head starts `data`: the head's length (through the blank line after it) and the request;
// nullopt while the head is not all there. Throws BadRequest: 400 for a malformed head, 431 for one longer than
// max_head bytes, 411 for a chunked body, 505 for a version other than HTTP/1.0 and 1.1.
std::optional<std::pair<size_t, Request>> parse_head(std::string_view data, size_t max_head = 16384);

// Percent-decoding, '+' a space where plus_is_space (a query's); throws BadRequest (400) for a bad escape.
std::string decode_url(std::string_view s, bool plus_is_space);

// The status's reason phrase ("OK", "Not Found", ...).
std::string_view reason(int status);

// A whole response: the status line, Content-Type, Content-Length, Connection, the extra headers, the body.
std::string response(int status, std::string_view content_type, std::string_view body, bool keep_alive,
                     const std::vector<std::pair<std::string, std::string>>& extra = {});

}  // namespace carbon::http
