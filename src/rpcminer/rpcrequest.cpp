/**
    Copyright (C) 2010  puddinpop

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/

#include "rpcrequest.h"
#include <sstream>
#include <cstring>
#include <vector>

RPCRequest::RPCRequest():m_url(""),m_user(""),m_password("")
{
	Init();
}

RPCRequest::RPCRequest(const std::string &url, const std::string &user, const std::string &password):m_url(url),m_user(user),m_password(password)
{
	Init();
}

RPCRequest::~RPCRequest()
{
	if(m_curl)
	{
		curl_easy_cleanup(m_curl);
	}
}

const bool RPCRequest::DoRequest(const std::string &data, std::string &result)
{
	int rval=-1;
	m_writebuff.clear();
	m_readbuff.assign(data.begin(),data.end());
	std::vector<std::string::value_type> vuserpass;

	if(m_curl)
	{
		struct curl_slist *headers=0;

		curl_easy_setopt(m_curl,CURLOPT_URL,m_url.c_str());
		curl_easy_setopt(m_curl,CURLOPT_ENCODING,"");
		curl_easy_setopt(m_curl,CURLOPT_FAILONERROR,1);
		curl_easy_setopt(m_curl,CURLOPT_TCP_NODELAY,1);
		curl_easy_setopt(m_curl,CURLOPT_WRITEFUNCTION,RPCRequest::WriteData);
		curl_easy_setopt(m_curl,CURLOPT_WRITEDATA,this);
		curl_easy_setopt(m_curl,CURLOPT_READFUNCTION,RPCRequest::ReadData);
		curl_easy_setopt(m_curl,CURLOPT_READDATA,this);
		if(m_user!="" && m_password!="")
		{
			std::string userpass(m_user+":"+m_password);
			// Apparenty, libCURL doesn't copy the string immediately, so scoped variables will not work,
			// so we'll use a vector declared previously in the method.
			vuserpass.assign(userpass.begin(),userpass.end());
			vuserpass.push_back(0);
			curl_easy_setopt(m_curl,CURLOPT_USERPWD,&vuserpass[0]);
			curl_easy_setopt(m_curl,CURLOPT_HTTPAUTH,CURLAUTH_BASIC);
		}
		curl_easy_setopt(m_curl,CURLOPT_POST,1);

		std::ostringstream istr;
		istr << m_readbuff.size();
		std::string headersize("Content-Length: "+istr.str());

		headers=curl_slist_append(headers,"Content-type: application/json");
		headers=curl_slist_append(headers,headersize.c_str());
		headers=curl_slist_append(headers,"Expect:");

		curl_easy_setopt(m_curl, CURLOPT_HTTPHEADER, headers);

		rval=curl_easy_perform(m_curl);
		m_lastcurlreturn=rval;
		
		if(m_writebuff.size()>0)
		{
			result.assign(m_writebuff.begin(),m_writebuff.end());
		}
		else
		{
			result="";
		}

		curl_slist_free_all(headers);

	}

	return (rval==0);
}

void RPCRequest::Init()
{
	m_lastcurlreturn=0;
	m_curl=curl_easy_init();
}

size_t RPCRequest::ReadData(void *ptr, size_t size, size_t nmemb, void *user_data)
{
	size_t readlen=size*nmemb;
	RPCRequest *req=(RPCRequest *)user_data;

	readlen=(std::min)(req->m_readbuff.size(),readlen);

	if(readlen>0)
	{
		::memcpy(ptr,&req->m_readbuff[0],readlen);

		req->m_readbuff.erase(req->m_readbuff.begin(),req->m_readbuff.begin()+readlen);
	}

	return readlen;
}

size_t RPCRequest::WriteData(void *ptr, size_t size, size_t nmemb, void *user_data)
{
	size_t writelen=size*nmemb;
	RPCRequest *req=(RPCRequest *)user_data;

	if(writelen>0)
	{
		std::vector<char>::size_type startpos=req->m_writebuff.size();
		req->m_writebuff.resize(req->m_writebuff.size()+writelen);
		::memcpy(&req->m_writebuff[startpos],ptr,writelen);
	}

	return writelen;
}
