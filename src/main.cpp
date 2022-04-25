//
// Created by 秋鱼头 on 2022/4/10.
//
#include <Agate/Core/Aplication.h>

using namespace Agate;

int main()
{
    Log log;
    
    AppProps props;
    
    Application app{props};

    app.Run();
}
