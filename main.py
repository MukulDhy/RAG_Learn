if __name__ == "__main__":
    from parsers.pdf_parser import parse_pdf

    result = parse_pdf("./policy1.pdf")
    print(result)
