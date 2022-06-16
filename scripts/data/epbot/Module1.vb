Imports System
Imports EPBot64.EPBot

Module Module1
    Private Structure TYPE_HAND
        Dim suit() As String
    End Structure

    Sub Main()
        Dim SEAT As New Dictionary(Of String, Integer)
        SEAT.Add("N", 0)
        SEAT.Add("E", 1)
        SEAT.Add("S", 2)
        SEAT.Add("W", 3)

        Dim VULN As New Dictionary(Of String, Integer)
        VULN.Add("None", 0)
        VULN.Add("E-W", 1)
        VULN.Add("N-S", 2)
        VULN.Add("Both", 3)

        Do
            Dim pass, dealer, current_bid, k, j, n, new_bid, position, vulnerability As Integer
            Dim meaning As String
            Dim info() As Integer
            Dim hand() As TYPE_HAND
            Dim Player() As EPBot64.EPBot
            Dim parts() As String

            Const C_PASS As Integer = 0
            Const C_REDOUBLE As Integer = 2
            Const C_CLUBS As Integer = 0
            Const C_DIAMONDS As Integer = 1
            Const C_HEARTS As Integer = 2
            Const C_SPADES As Integer = 3
            Const C_NS As Integer = 0
            Const C_WE As Integer = 1
            Const C_IMP As Integer = 1

            '0 = 2/1
            '1 = sayc
            '2 = polish club
            '3 = precision
            Dim bidding_system = 0

            ReDim Player(3)

            For k = 0 To 3
                Player(k) = New EPBot64.EPBot
            Next k

            ReDim hand(3)

            For k = 0 To 3
                ReDim hand(k).suit(3)
            Next

            parts = Console.ReadLine().Trim().Split()
            Console.WriteLine(String.Join(" ", parts))

            For k = 2 To 5
                Dim suits() As String
                suits = parts(k).Split("."c)
                hand(k - 2).suit(C_CLUBS) = suits(3)
                hand(k - 2).suit(C_DIAMONDS) = suits(2)
                hand(k - 2).suit(C_HEARTS) = suits(1)
                hand(k - 2).suit(C_SPADES) = suits(0)
            Next

            dealer = SEAT(parts(0))
            vulnerability = VULN(parts(1))

            For position = 0 To 3
                Player(4 * n + position).scoring = C_IMP

                Player(4 * n + position).system_type(C_NS) = bidding_system

                Player(4 * n + position).system_type(C_WE) = bidding_system

                Player(4 * n + position).new_hand(position, hand(position).suit, dealer, vulnerability)

            Next position

            pass = 0

            new_bid = 0

            current_bid = 0

            k = 4 * n + dealer


            Do

                position = k Mod 4

                new_bid = Player(k).get_bid

                Console.WriteLine("BID = " + new_bid.ToString())

                If new_bid = C_PASS Then

                    pass = pass + 1

                Else

                    pass = 0

                    If new_bid > C_REDOUBLE Then

                        current_bid = new_bid

                    End If

                End If

                For j = 4 * n To 4 * n + 3

                    Player(j).set_bid(position, new_bid)

                Next j

                With Player(k)

                    '---get info from Player(k) about k player

                    meaning = .info_meaning(position)
                    Console.WriteLine("meaning: " + .info_meaning(position))


                    info = .info_feature(position)

                    info = .info_honors(position)

                    info = .info_max_length(position)
                    Console.WriteLine("max-length: " + String.Join(",", info.Select(Function(x) x.ToString()).ToArray()))

                    info = .info_min_length(position)
                    Console.WriteLine("min-length: " + String.Join(",", info.Select(Function(x) x.ToString()).ToArray()))

                    info = .info_probable_length(position)

                    info = .info_suit_power(position)



                    info = .info_stoppers(position)


                End With

                '---set a new player

                k = 4 * n + (k + 1) Mod 4

            Loop While (pass < 3 Or pass < 4 And current_bid = 0)

        Loop

    End Sub

End Module
