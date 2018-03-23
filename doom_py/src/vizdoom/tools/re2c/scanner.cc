/* Generated by re2c 0.12.3 on Fri Jan 25 21:09:48 2008 */
/* $Id: scanner.re 663 2007-04-01 11:22:15Z helly $ */
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include "scanner.h"
#include "parser.h"
#include "y.tab.h"
#include "globals.h"
#include "dfa.h"

extern YYSTYPE yylval;

#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

#define	BSIZE	8192

#define	YYCTYPE		unsigned char
#define	YYCURSOR	cursor
#define	YYLIMIT		lim
#define	YYMARKER	ptr
#define	YYFILL(n)	{cursor = fill(cursor);}

#define	RETURN(i)	{cur = cursor; return i;}

namespace re2c
{

Scanner::Scanner(const char *fn, std::istream& i, std::ostream& o)
	: in(i)
	, out(o)
	, bot(NULL), tok(NULL), ptr(NULL), cur(NULL), pos(NULL), lim(NULL)
	, top(NULL), eof(NULL), tchar(0), tline(0), cline(1), iscfg(0), filename(fn)
{
    ;
}

char *Scanner::fill(char *cursor)
{
	if(!eof)
	{
		uint cnt = tok - bot;
		if(cnt)
		{
			memcpy(bot, tok, lim - tok);
			tok = bot;
			ptr -= cnt;
			cursor -= cnt;
			pos -= cnt;
			lim -= cnt;
		}
		if((top - lim) < BSIZE)
		{
			char *buf = new char[(lim - bot) + BSIZE];
			memcpy(buf, tok, lim - tok);
			tok = buf;
			ptr = &buf[ptr - bot];
			cursor = &buf[cursor - bot];
			pos = &buf[pos - bot];
			lim = &buf[lim - bot];
			top = &lim[BSIZE];
			delete [] bot;
			bot = buf;
		}
		in.read(lim, BSIZE);
		if ((cnt = in.gcount()) != BSIZE )
		{
			eof = &lim[cnt]; *eof++ = '\0';
		}
		lim += cnt;
	}
	return cursor;
}



int Scanner::echo()
{
    char *cursor = cur;
    bool ignore_eoc = false;
    int  ignore_cnt = 0;

    if (eof && cursor == eof) // Catch EOF
	{
    	return 0;
	}

    tok = cursor;
echo:
{

	{
		YYCTYPE yych;
		unsigned int yyaccept = 0;

		if((YYLIMIT - YYCURSOR) < 16) YYFILL(16);
		yych = *YYCURSOR;
		if(yych <= ')') {
			if(yych <= 0x00) goto yy7;
			if(yych == 0x0A) goto yy5;
			goto yy9;
		} else {
			if(yych <= '*') goto yy4;
			if(yych != '/') goto yy9;
		}
		yyaccept = 0;
		yych = *(YYMARKER = ++YYCURSOR);
		if(yych == '*') goto yy16;
yy3:
		{
					goto echo;
				}
yy4:
		yych = *++YYCURSOR;
		if(yych == '/') goto yy10;
		goto yy3;
yy5:
		++YYCURSOR;
		{
					if (ignore_eoc) {
						ignore_cnt++;
					} else {
						out.write((const char*)(tok), (const char*)(cursor) - (const char*)(tok));
					}
					tok = pos = cursor; cline++;
				  	goto echo;
				}
yy7:
		++YYCURSOR;
		{
					if (!ignore_eoc) {
						out.write((const char*)(tok), (const char*)(cursor) - (const char*)(tok) - 1); // -1 so we don't write out the \0
					}
					if(cursor == eof) {
						RETURN(0);
					}
				}
yy9:
		yych = *++YYCURSOR;
		goto yy3;
yy10:
		yyaccept = 1;
		yych = *(YYMARKER = ++YYCURSOR);
		if(yych == 0x0A) goto yy14;
		if(yych == 0x0D) goto yy12;
yy11:
		{
					if (ignore_eoc) {
						if (ignore_cnt) {
							out << "\n" << sourceFileInfo;
						}
						ignore_eoc = false;
						ignore_cnt = 0;
					} else {
						out.write((const char*)(tok), (const char*)(cursor) - (const char*)(tok));
					}
					tok = pos = cursor;
					goto echo;
				}
yy12:
		yych = *++YYCURSOR;
		if(yych == 0x0A) goto yy14;
yy13:
		YYCURSOR = YYMARKER;
		if(yyaccept <= 0) {
			goto yy3;
		} else {
			goto yy11;
		}
yy14:
		++YYCURSOR;
		{
					cline++;
					if (ignore_eoc) {
						if (ignore_cnt) {
							out << sourceFileInfo;
						}
						ignore_eoc = false;
						ignore_cnt = 0;
					} else {
						out.write((const char*)(tok), (const char*)(cursor) - (const char*)(tok));
					}
					tok = pos = cursor;
					goto echo;
				}
yy16:
		yych = *++YYCURSOR;
		if(yych != '!') goto yy13;
		yych = *++YYCURSOR;
		switch(yych) {
		case 'g':	goto yy19;
		case 'i':	goto yy18;
		case 'm':	goto yy20;
		case 'r':	goto yy21;
		default:	goto yy13;
		}
yy18:
		yych = *++YYCURSOR;
		if(yych == 'g') goto yy47;
		goto yy13;
yy19:
		yych = *++YYCURSOR;
		if(yych == 'e') goto yy34;
		goto yy13;
yy20:
		yych = *++YYCURSOR;
		if(yych == 'a') goto yy26;
		goto yy13;
yy21:
		yych = *++YYCURSOR;
		if(yych != 'e') goto yy13;
		yych = *++YYCURSOR;
		if(yych != '2') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 'c') goto yy13;
		++YYCURSOR;
		{
					if (bUsedYYMaxFill && bSinglePass) {
						fatal("found scanner block after YYMAXFILL declaration");
					}
					out.write((const char*)(tok), (const char*)(&cursor[-7]) - (const char*)(tok));
					tok = cursor;
					RETURN(1);
				}
yy26:
		yych = *++YYCURSOR;
		if(yych != 'x') goto yy13;
		yych = *++YYCURSOR;
		if(yych != ':') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 'r') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 'e') goto yy13;
		yych = *++YYCURSOR;
		if(yych != '2') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 'c') goto yy13;
		++YYCURSOR;
		{
					if (bUsedYYMaxFill) {
						fatal("cannot generate YYMAXFILL twice");
					}
					out << "#define YYMAXFILL " << maxFill << std::endl;
					tok = pos = cursor;
					ignore_eoc = true;
					bUsedYYMaxFill = true;
					goto echo;
				}
yy34:
		yych = *++YYCURSOR;
		if(yych != 't') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 's') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 't') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 'a') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 't') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 'e') goto yy13;
		yych = *++YYCURSOR;
		if(yych != ':') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 'r') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 'e') goto yy13;
		yych = *++YYCURSOR;
		if(yych != '2') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 'c') goto yy13;
		++YYCURSOR;
		{
					tok = pos = cursor;
					genGetState(out, topIndent, 0);
					ignore_eoc = true;
					goto echo;
				}
yy47:
		yych = *++YYCURSOR;
		if(yych != 'n') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 'o') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 'r') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 'e') goto yy13;
		yych = *++YYCURSOR;
		if(yych != ':') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 'r') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 'e') goto yy13;
		yych = *++YYCURSOR;
		if(yych != '2') goto yy13;
		yych = *++YYCURSOR;
		if(yych != 'c') goto yy13;
		++YYCURSOR;
		{
					tok = pos = cursor;
					ignore_eoc = true;
					goto echo;
				}
	}
}

}


int Scanner::scan()
{
    char *cursor = cur;
    uint depth;

scan:
    tchar = cursor - pos;
    tline = cline;
    tok = cursor;
	if (iscfg == 1)
	{
		goto config;
	}
	else if (iscfg == 2)
	{
   		goto value;
    }
{
	static const unsigned char yybm[] = {
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 116,   0, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		116, 112,  48, 112, 112, 112, 112,  80, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 112, 112, 112, 112, 112, 112, 
		112, 120, 120, 120, 120, 120, 120, 120, 
		120, 120, 120, 120, 120, 120, 120, 120, 
		120, 120, 120, 120, 120, 120, 120, 120, 
		120, 120, 120, 112,   0,  96, 112, 120, 
		112, 120, 120, 120, 120, 120, 120, 120, 
		120, 120, 120, 120, 120, 120, 120, 120, 
		120, 120, 120, 120, 120, 120, 120, 120, 
		120, 120, 120, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
		112, 112, 112, 112, 112, 112, 112, 112, 
	};

	{
		YYCTYPE yych;
		unsigned int yyaccept = 0;
		if((YYLIMIT - YYCURSOR) < 5) YYFILL(5);
		yych = *YYCURSOR;
		if(yych <= ':') {
			if(yych <= '"') {
				if(yych <= 0x0C) {
					if(yych <= 0x08) goto yy85;
					if(yych <= 0x09) goto yy79;
					if(yych <= 0x0A) goto yy81;
					goto yy85;
				} else {
					if(yych <= 0x1F) {
						if(yych <= 0x0D) goto yy83;
						goto yy85;
					} else {
						if(yych <= ' ') goto yy79;
						if(yych <= '!') goto yy85;
						goto yy66;
					}
				}
			} else {
				if(yych <= '*') {
					if(yych <= '&') goto yy85;
					if(yych <= '\'') goto yy68;
					if(yych <= ')') goto yy72;
					goto yy64;
				} else {
					if(yych <= '-') {
						if(yych <= '+') goto yy73;
						goto yy85;
					} else {
						if(yych <= '.') goto yy77;
						if(yych <= '/') goto yy62;
						goto yy85;
					}
				}
			}
		} else {
			if(yych <= '\\') {
				if(yych <= '>') {
					if(yych == '<') goto yy85;
					if(yych <= '=') goto yy72;
					goto yy85;
				} else {
					if(yych <= '@') {
						if(yych <= '?') goto yy73;
						goto yy85;
					} else {
						if(yych <= 'Z') goto yy76;
						if(yych <= '[') goto yy70;
						goto yy72;
					}
				}
			} else {
				if(yych <= 'q') {
					if(yych == '_') goto yy76;
					if(yych <= '`') goto yy85;
					goto yy76;
				} else {
					if(yych <= 'z') {
						if(yych <= 'r') goto yy74;
						goto yy76;
					} else {
						if(yych <= '{') goto yy60;
						if(yych <= '|') goto yy72;
						goto yy85;
					}
				}
			}
		}
yy60:
		yyaccept = 0;
		yych = *(YYMARKER = ++YYCURSOR);
		if(yych <= '/') {
			if(yych == ',') goto yy126;
		} else {
			if(yych <= '0') goto yy123;
			if(yych <= '9') goto yy124;
		}
yy61:
		{ depth = 1;
				  goto code;
				}
yy62:
		++YYCURSOR;
		if((yych = *YYCURSOR) == '*') goto yy121;
yy63:
		{ RETURN(*tok); }
yy64:
		++YYCURSOR;
		if((yych = *YYCURSOR) == '/') goto yy119;
yy65:
		{ yylval.op = *tok;
				  RETURN(CLOSE); }
yy66:
		yyaccept = 1;
		yych = *(YYMARKER = ++YYCURSOR);
		if(yych != 0x0A) goto yy115;
yy67:
		{ fatal("unterminated string constant (missing \")"); }
yy68:
		yyaccept = 2;
		yych = *(YYMARKER = ++YYCURSOR);
		if(yych != 0x0A) goto yy110;
yy69:
		{ fatal("unterminated string constant (missing ')"); }
yy70:
		yyaccept = 3;
		yych = *(YYMARKER = ++YYCURSOR);
		if(yych == 0x0A) goto yy71;
		if(yych == '^') goto yy101;
		goto yy100;
yy71:
		{ fatal("unterminated range (missing ])"); }
yy72:
		yych = *++YYCURSOR;
		goto yy63;
yy73:
		yych = *++YYCURSOR;
		goto yy65;
yy74:
		++YYCURSOR;
		if((yych = *YYCURSOR) == 'e') goto yy91;
		goto yy90;
yy75:
		{ cur = cursor;
				  yylval.symbol = Symbol::find(token());
				  return ID; }
yy76:
		yych = *++YYCURSOR;
		goto yy90;
yy77:
		++YYCURSOR;
		{ cur = cursor;
				  yylval.regexp = mkDot();
				  return RANGE;
				}
yy79:
		++YYCURSOR;
		yych = *YYCURSOR;
		goto yy88;
yy80:
		{ goto scan; }
yy81:
		++YYCURSOR;
yy82:
		{ if(cursor == eof) RETURN(0);
				  pos = cursor; cline++;
				  goto scan;
	    			}
yy83:
		++YYCURSOR;
		if((yych = *YYCURSOR) == 0x0A) goto yy86;
yy84:
		{ std::ostringstream msg;
				  msg << "unexpected character: ";
				  prtChOrHex(msg, *tok);
				  fatal(msg.str().c_str());
				  goto scan;
				}
yy85:
		yych = *++YYCURSOR;
		goto yy84;
yy86:
		yych = *++YYCURSOR;
		goto yy82;
yy87:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
yy88:
		if(yybm[0+yych] & 4) {
			goto yy87;
		}
		goto yy80;
yy89:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
yy90:
		if(yybm[0+yych] & 8) {
			goto yy89;
		}
		goto yy75;
yy91:
		yych = *++YYCURSOR;
		if(yych != '2') goto yy90;
		yych = *++YYCURSOR;
		if(yych != 'c') goto yy90;
		yyaccept = 4;
		yych = *(YYMARKER = ++YYCURSOR);
		if(yych != ':') goto yy90;
yy94:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yych <= '^') {
			if(yych <= '@') goto yy95;
			if(yych <= 'Z') goto yy96;
		} else {
			if(yych == '`') goto yy95;
			if(yych <= 'z') goto yy96;
		}
yy95:
		YYCURSOR = YYMARKER;
		if(yyaccept <= 3) {
			if(yyaccept <= 1) {
				if(yyaccept <= 0) {
					goto yy61;
				} else {
					goto yy67;
				}
			} else {
				if(yyaccept <= 2) {
					goto yy69;
				} else {
					goto yy71;
				}
			}
		} else {
			if(yyaccept <= 5) {
				if(yyaccept <= 4) {
					goto yy75;
				} else {
					goto yy98;
				}
			} else {
				goto yy127;
			}
		}
yy96:
		yyaccept = 5;
		YYMARKER = ++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yych <= 'Z') {
			if(yych <= '9') {
				if(yych >= '0') goto yy96;
			} else {
				if(yych <= ':') goto yy94;
				if(yych >= 'A') goto yy96;
			}
		} else {
			if(yych <= '_') {
				if(yych >= '_') goto yy96;
			} else {
				if(yych <= '`') goto yy98;
				if(yych <= 'z') goto yy96;
			}
		}
yy98:
		{ cur = cursor;
				  tok+= 5; /* skip "re2c:" */
				  iscfg = 1;
				  yylval.str = new Str(token());
				  return CONFIG;
				}
yy99:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
yy100:
		if(yybm[0+yych] & 16) {
			goto yy99;
		}
		if(yych <= '[') goto yy95;
		if(yych <= '\\') goto yy103;
		goto yy104;
yy101:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yych <= '[') {
			if(yych == 0x0A) goto yy95;
			goto yy101;
		} else {
			if(yych <= '\\') goto yy106;
			if(yych <= ']') goto yy107;
			goto yy101;
		}
yy103:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yych == 0x0A) goto yy95;
		goto yy99;
yy104:
		++YYCURSOR;
		{ cur = cursor;
				  yylval.regexp = ranToRE(token());
				  return RANGE; }
yy106:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yych == 0x0A) goto yy95;
		goto yy101;
yy107:
		++YYCURSOR;
		{ cur = cursor;
				  yylval.regexp = invToRE(token());
				  return RANGE; }
yy109:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
yy110:
		if(yybm[0+yych] & 32) {
			goto yy109;
		}
		if(yych <= '&') goto yy95;
		if(yych <= '\'') goto yy112;
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yych == 0x0A) goto yy95;
		goto yy109;
yy112:
		++YYCURSOR;
		{ cur = cursor;
				  yylval.regexp = strToCaseInsensitiveRE(token());
				  return STRING; }
yy114:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
yy115:
		if(yybm[0+yych] & 64) {
			goto yy114;
		}
		if(yych <= '!') goto yy95;
		if(yych <= '"') goto yy117;
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yych == 0x0A) goto yy95;
		goto yy114;
yy117:
		++YYCURSOR;
		{ cur = cursor;
				  yylval.regexp = strToRE(token());
				  return STRING; }
yy119:
		++YYCURSOR;
		{ tok = cursor;
				  RETURN(0); }
yy121:
		++YYCURSOR;
		{ depth = 1;
				  goto comment; }
yy123:
		yych = *++YYCURSOR;
		if(yych == ',') goto yy137;
		goto yy125;
yy124:
		++YYCURSOR;
		if((YYLIMIT - YYCURSOR) < 2) YYFILL(2);
		yych = *YYCURSOR;
yy125:
		if(yybm[0+yych] & 128) {
			goto yy124;
		}
		if(yych == ',') goto yy130;
		if(yych == '}') goto yy128;
		goto yy95;
yy126:
		++YYCURSOR;
yy127:
		{ fatal("illegal closure form, use '{n}', '{n,}', '{n,m}' where n and m are numbers"); }
yy128:
		++YYCURSOR;
		{ yylval.extop.minsize = atoi((char *)tok+1);
				  yylval.extop.maxsize = atoi((char *)tok+1);
				  RETURN(CLOSESIZE); }
yy130:
		yyaccept = 6;
		yych = *(YYMARKER = ++YYCURSOR);
		if(yych <= '/') goto yy127;
		if(yych <= '9') goto yy133;
		if(yych != '}') goto yy127;
		++YYCURSOR;
		{ yylval.extop.minsize = atoi((char *)tok+1);
				  yylval.extop.maxsize = -1;
				  RETURN(CLOSESIZE); }
yy133:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yych <= '/') goto yy95;
		if(yych <= '9') goto yy133;
		if(yych != '}') goto yy95;
		++YYCURSOR;
		{ yylval.extop.minsize = atoi((char *)tok+1);
				  yylval.extop.maxsize = MAX(yylval.extop.minsize,atoi(strchr((char *)tok, ',')+1));
				  RETURN(CLOSESIZE); }
yy137:
		yyaccept = 6;
		yych = *(YYMARKER = ++YYCURSOR);
		if(yych <= '/') goto yy127;
		if(yych <= '9') goto yy133;
		if(yych != '}') goto yy127;
		++YYCURSOR;
		{ yylval.op = '*';
				  RETURN(CLOSE); }
	}
}


code:
{
	static const unsigned char yybm[] = {
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192,   0, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192,  64, 192, 192, 192, 192, 128, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192,   0, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
		192, 192, 192, 192, 192, 192, 192, 192, 
	};

	{
		YYCTYPE yych;
		if((YYLIMIT - YYCURSOR) < 2) YYFILL(2);
		yych = *YYCURSOR;
		if(yych <= '&') {
			if(yych <= 0x0A) {
				if(yych <= 0x00) goto yy148;
				if(yych <= 0x09) goto yy150;
				goto yy146;
			} else {
				if(yych == '"') goto yy152;
				goto yy150;
			}
		} else {
			if(yych <= '{') {
				if(yych <= '\'') goto yy153;
				if(yych <= 'z') goto yy150;
				goto yy144;
			} else {
				if(yych != '}') goto yy150;
			}
		}
		++YYCURSOR;
		{ if(--depth == 0){
					cur = cursor;
					yylval.token = new Token(token(), tline);
					return CODE;
				  }
				  goto code; }
yy144:
		++YYCURSOR;
		{ ++depth;
				  goto code; }
yy146:
		++YYCURSOR;
		{ if(cursor == eof) fatal("missing '}'");
				  pos = cursor; cline++;
				  goto code;
				}
yy148:
		++YYCURSOR;
		{ if(cursor == eof) {
					if (depth) fatal("missing '}'");
					RETURN(0);
				  }
				  goto code;
				}
yy150:
		++YYCURSOR;
yy151:
		{ goto code; }
yy152:
		yych = *(YYMARKER = ++YYCURSOR);
		if(yych == 0x0A) goto yy151;
		goto yy159;
yy153:
		yych = *(YYMARKER = ++YYCURSOR);
		if(yych == 0x0A) goto yy151;
		goto yy155;
yy154:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
yy155:
		if(yybm[0+yych] & 64) {
			goto yy154;
		}
		if(yych <= '&') goto yy156;
		if(yych <= '\'') goto yy150;
		goto yy157;
yy156:
		YYCURSOR = YYMARKER;
		goto yy151;
yy157:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yych == 0x0A) goto yy156;
		goto yy154;
yy158:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
yy159:
		if(yybm[0+yych] & 128) {
			goto yy158;
		}
		if(yych <= '!') goto yy156;
		if(yych <= '"') goto yy150;
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yych == 0x0A) goto yy156;
		goto yy158;
	}
}


comment:
{

	{
		YYCTYPE yych;
		if((YYLIMIT - YYCURSOR) < 2) YYFILL(2);
		yych = *YYCURSOR;
		if(yych <= ')') {
			if(yych == 0x0A) goto yy166;
			goto yy168;
		} else {
			if(yych <= '*') goto yy163;
			if(yych == '/') goto yy165;
			goto yy168;
		}
yy163:
		++YYCURSOR;
		if((yych = *YYCURSOR) == '/') goto yy171;
yy164:
		{ if(cursor == eof) RETURN(0);
				  goto comment; }
yy165:
		yych = *++YYCURSOR;
		if(yych == '*') goto yy169;
		goto yy164;
yy166:
		++YYCURSOR;
		{ if(cursor == eof) RETURN(0);
				  tok = pos = cursor; cline++;
				  goto comment;
				}
yy168:
		yych = *++YYCURSOR;
		goto yy164;
yy169:
		++YYCURSOR;
		{ ++depth;
				  fatal("ambiguous /* found");
				  goto comment; }
yy171:
		++YYCURSOR;
		{ if(--depth == 0)
					goto scan;
				    else
					goto comment; }
	}
}


config:
{
	static const unsigned char yybm[] = {
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0, 128,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		128,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
		  0,   0,   0,   0,   0,   0,   0,   0, 
	};

	{
		YYCTYPE yych;
		if((YYLIMIT - YYCURSOR) < 2) YYFILL(2);
		yych = *YYCURSOR;
		if(yych <= 0x1F) {
			if(yych != 0x09) goto yy179;
		} else {
			if(yych <= ' ') goto yy175;
			if(yych == '=') goto yy177;
			goto yy179;
		}
yy175:
		++YYCURSOR;
		yych = *YYCURSOR;
		goto yy184;
yy176:
		{ goto config; }
yy177:
		++YYCURSOR;
		yych = *YYCURSOR;
		goto yy182;
yy178:
		{ iscfg = 2;
				  cur = cursor;
				  RETURN('='); 
				}
yy179:
		++YYCURSOR;
		{ fatal("missing '='"); }
yy181:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
yy182:
		if(yybm[0+yych] & 128) {
			goto yy181;
		}
		goto yy178;
yy183:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
yy184:
		if(yych == 0x09) goto yy183;
		if(yych == ' ') goto yy183;
		goto yy176;
	}
}


value:
{
	static const unsigned char yybm[] = {
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 192,   0, 248, 248, 192, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		192, 248, 104, 248, 248, 248, 248, 152, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		252, 252, 252, 252, 252, 252, 252, 252, 
		252, 252, 248, 192, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248,   8, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
		248, 248, 248, 248, 248, 248, 248, 248, 
	};

	{
		YYCTYPE yych;
		if((YYLIMIT - YYCURSOR) < 2) YYFILL(2);
		yych = *YYCURSOR;
		if(yych <= '&') {
			if(yych <= 0x0D) {
				if(yych <= 0x08) goto yy193;
				if(yych <= 0x0A) goto yy187;
				if(yych <= 0x0C) goto yy193;
			} else {
				if(yych <= ' ') {
					if(yych <= 0x1F) goto yy193;
				} else {
					if(yych == '"') goto yy195;
					goto yy193;
				}
			}
		} else {
			if(yych <= '/') {
				if(yych <= '\'') goto yy197;
				if(yych == '-') goto yy190;
				goto yy193;
			} else {
				if(yych <= '9') {
					if(yych <= '0') goto yy188;
					goto yy191;
				} else {
					if(yych != ';') goto yy193;
				}
			}
		}
yy187:
		{ cur = cursor;
				  yylval.str = new Str(token());
				  iscfg = 0;
				  return VALUE;
				}
yy188:
		++YYCURSOR;
		if(yybm[0+(yych = *YYCURSOR)] & 8) {
			goto yy193;
		}
yy189:
		{ cur = cursor;
				  yylval.number = atoi(token().to_string().c_str());
				  iscfg = 0;
				  return NUMBER;
				}
yy190:
		yych = *++YYCURSOR;
		if(yych <= '0') goto yy194;
		if(yych >= ':') goto yy194;
yy191:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yybm[0+yych] & 4) {
			goto yy191;
		}
		if(yych <= 0x0D) {
			if(yych <= 0x08) goto yy193;
			if(yych <= 0x0A) goto yy189;
			if(yych >= 0x0D) goto yy189;
		} else {
			if(yych <= ' ') {
				if(yych >= ' ') goto yy189;
			} else {
				if(yych == ';') goto yy189;
			}
		}
yy193:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
yy194:
		if(yybm[0+yych] & 8) {
			goto yy193;
		}
		goto yy187;
yy195:
		YYMARKER = ++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yybm[0+yych] & 16) {
			goto yy195;
		}
		if(yych <= '!') {
			if(yych == 0x0A) goto yy187;
			goto yy205;
		} else {
			if(yych <= '"') goto yy193;
			if(yych <= '[') goto yy205;
			goto yy207;
		}
yy197:
		YYMARKER = ++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yybm[0+yych] & 32) {
			goto yy197;
		}
		if(yych <= '&') {
			if(yych == 0x0A) goto yy187;
		} else {
			if(yych <= '\'') goto yy193;
			if(yych >= '\\') goto yy202;
		}
yy199:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yybm[0+yych] & 64) {
			goto yy199;
		}
		if(yych <= '&') goto yy201;
		if(yych <= '\'') goto yy203;
		goto yy204;
yy201:
		YYCURSOR = YYMARKER;
		goto yy187;
yy202:
		YYMARKER = ++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yych <= 0x0D) {
			if(yych <= 0x09) {
				if(yych <= 0x08) goto yy197;
				goto yy199;
			} else {
				if(yych <= 0x0A) goto yy187;
				if(yych <= 0x0C) goto yy197;
				goto yy199;
			}
		} else {
			if(yych <= ' ') {
				if(yych <= 0x1F) goto yy197;
				goto yy199;
			} else {
				if(yych == ';') goto yy199;
				goto yy197;
			}
		}
yy203:
		yych = *++YYCURSOR;
		goto yy187;
yy204:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yych == 0x0A) goto yy201;
		goto yy199;
yy205:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yybm[0+yych] & 128) {
			goto yy205;
		}
		if(yych <= '!') goto yy201;
		if(yych <= '"') goto yy203;
		goto yy208;
yy207:
		YYMARKER = ++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yych <= 0x0D) {
			if(yych <= 0x09) {
				if(yych <= 0x08) goto yy195;
				goto yy205;
			} else {
				if(yych <= 0x0A) goto yy187;
				if(yych <= 0x0C) goto yy195;
				goto yy205;
			}
		} else {
			if(yych <= ' ') {
				if(yych <= 0x1F) goto yy195;
				goto yy205;
			} else {
				if(yych == ';') goto yy205;
				goto yy195;
			}
		}
yy208:
		++YYCURSOR;
		if(YYLIMIT == YYCURSOR) YYFILL(1);
		yych = *YYCURSOR;
		if(yych == 0x0A) goto yy201;
		goto yy205;
	}
}

}

void Scanner::fatal(uint ofs, const char *msg) const
{
	out.flush();
#ifdef _MSC_VER
	std::cerr << filename << "(" << tline << "): error : "
		<< "column " << (tchar + ofs + 1) << ": "
		<< msg << std::endl;
#else
	std::cerr << "re2c: error: "
		<< "line " << tline << ", column " << (tchar + ofs + 1) << ": "
		<< msg << std::endl;
#endif
   	exit(1);
}

Scanner::~Scanner()
{
	if (bot)
	{
		delete [] bot;
	}
}

} // end namespace re2c

