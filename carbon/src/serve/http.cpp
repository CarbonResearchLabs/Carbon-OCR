#include "carbon/http.h"

#include <cctype>
#include <charconv>

namespace carbon::http {

namespace {

constexpr std::string_view kCrlf = "\r\n";

std::string lower(std::string_view s) {
  std::string out(s);
  for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

std::string_view trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
  return s;
}

// A token's characters (RFC 9110 5.6.2): a method's and a header name's.
bool is_token(std::string_view s) {
  if (s.empty()) return false;
  for (char ch : s) {
    const unsigned char c = static_cast<unsigned char>(ch);
    if (std::isalnum(c)) continue;
    if (std::string_view("!#$%&'*+-.^_`|~").find(static_cast<char>(c)) == std::string_view::npos) return false;
  }
  return true;
}

// Whether a comma-separated header value (Connection's, Transfer-Encoding's) names `token`, in any case.
bool lists(std::string_view value, std::string_view token) {
  while (!value.empty()) {
    const size_t comma = value.find(',');
    if (lower(trim(value.substr(0, comma))) == token) return true;
    if (comma == std::string_view::npos) break;
    value.remove_prefix(comma + 1);
  }
  return false;
}

int hex(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

}  // namespace

std::optional<std::string_view> Request::header(std::string_view lower_name) const {
  for (const auto& [k, v] : headers) {
    if (k == lower_name) return std::string_view(v);
  }
  return std::nullopt;
}

std::optional<std::string_view> Request::param(std::string_view name) const {
  for (const auto& [k, v] : query) {
    if (k == name) return std::string_view(v);
  }
  return std::nullopt;
}

std::string decode_url(std::string_view s, bool plus_is_space) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%') {
      const int a = i + 2 < s.size() ? hex(s[i + 1]) : -1;
      const int b = a < 0 ? -1 : hex(s[i + 2]);
      if (b < 0) throw BadRequest(400, "a bad percent escape in the target");
      out += static_cast<char>(a * 16 + b);
      i += 2;
    } else if (s[i] == '+' && plus_is_space) {
      out += ' ';
    } else {
      out += s[i];
    }
  }
  return out;
}

std::optional<std::pair<size_t, Request>> parse_head(std::string_view data, size_t max_head) {
  const size_t end = data.find("\r\n\r\n");
  if (end == std::string_view::npos) {
    if (data.size() > max_head) throw BadRequest(431, "the request's head is longer than " +
                                                          std::to_string(max_head) + " bytes");
    return std::nullopt;
  }
  if (end + 4 > max_head) {
    throw BadRequest(431, "the request's head is longer than " + std::to_string(max_head) + " bytes");
  }
  std::string_view head = data.substr(0, end + 2);  // every line with its CRLF
  Request r;

  // The request line: METHOD SP target SP HTTP/x.y
  size_t eol = head.find(kCrlf);
  const std::string_view line = head.substr(0, eol);
  head.remove_prefix(eol + 2);
  const size_t sp1 = line.find(' ');
  const size_t sp2 = sp1 == std::string_view::npos ? sp1 : line.find(' ', sp1 + 1);
  if (sp2 == std::string_view::npos || line.find(' ', sp2 + 1) != std::string_view::npos) {
    throw BadRequest(400, "a malformed request line");
  }
  r.method = std::string(line.substr(0, sp1));
  r.target = std::string(line.substr(sp1 + 1, sp2 - sp1 - 1));
  r.version = std::string(line.substr(sp2 + 1));
  if (!is_token(r.method) || r.target.empty() || r.target.front() != '/') {
    throw BadRequest(400, "a malformed request line");
  }
  if (r.version != "HTTP/1.1" && r.version != "HTTP/1.0") {
    if (r.version.rfind("HTTP/", 0) != 0) throw BadRequest(400, "a malformed request line");
    throw BadRequest(505, "HTTP/1.0 and HTTP/1.1 only, not " + r.version);
  }
  const size_t q = r.target.find('?');
  r.path = decode_url(std::string_view(r.target).substr(0, q), false);
  if (q != std::string::npos) {
    std::string_view rest = std::string_view(r.target).substr(q + 1);
    while (!rest.empty()) {
      const size_t amp = rest.find('&');
      const std::string_view pair = rest.substr(0, amp);
      if (!pair.empty()) {
        const size_t eq = pair.find('=');
        r.query.emplace_back(decode_url(pair.substr(0, eq), true),
                             eq == std::string_view::npos ? std::string() : decode_url(pair.substr(eq + 1), true));
      }
      if (amp == std::string_view::npos) break;
      rest.remove_prefix(amp + 1);
    }
  }

  // The headers: name ":" OWS value OWS, a line each (no obsolete line folding)
  while (!head.empty()) {
    eol = head.find(kCrlf);
    const std::string_view h = head.substr(0, eol);
    head.remove_prefix(eol + 2);
    const size_t colon = h.find(':');
    if (colon == std::string_view::npos || !is_token(h.substr(0, colon))) throw BadRequest(400, "a malformed header");
    r.headers.emplace_back(lower(h.substr(0, colon)), std::string(trim(h.substr(colon + 1))));
  }
  for (const auto& [k, v] : r.headers) {
    if (k == "content-length") {
      size_t n = 0;
      const auto [p, ec] = std::from_chars(v.data(), v.data() + v.size(), n);
      if (ec != std::errc() || p != v.data() + v.size() || v.empty()) throw BadRequest(400, "a bad Content-Length");
      if (r.has_length && n != r.content_length) throw BadRequest(400, "two Content-Lengths that differ");
      r.has_length = true;
      r.content_length = n;
    } else if (k == "transfer-encoding") {
      throw BadRequest(411, "a body by its Content-Length only, not a Transfer-Encoding (" + v + ")");
    } else if (k == "expect") {
      if (lower(v) != "100-continue") throw BadRequest(417, "Expect: " + v);
      r.expect_continue = true;
    }
  }
  const auto connection = r.header("connection");
  r.keep_alive = r.version == "HTTP/1.1" ? !(connection && lists(*connection, "close"))
                                         : (connection && lists(*connection, "keep-alive"));
  return std::make_pair(end + 4, std::move(r));
}

std::string_view reason(int status) {
  switch (status) {
    case 100: return "Continue";
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 411: return "Length Required";
    case 413: return "Content Too Large";
    case 415: return "Unsupported Media Type";
    case 417: return "Expectation Failed";
    case 422: return "Unprocessable Content";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    case 505: return "HTTP Version Not Supported";
    default: return "Unknown";
  }
}

std::string response(int status, std::string_view content_type, std::string_view body, bool keep_alive,
                     const std::vector<std::pair<std::string, std::string>>& extra) {
  std::string out = "HTTP/1.1 " + std::to_string(status) + " " + std::string(reason(status)) + "\r\n";
  out += "Content-Type: ";
  out += content_type;
  out += "\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
  out += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
  for (const auto& [k, v] : extra) out += k + ": " + v + "\r\n";
  out += "\r\n";
  out += body;
  return out;
}

}  // namespace carbon::http
